"""Image-to-text captioning for Halcyon models with vLLM.

Outputs under ``--out-dir`` (default ``outputs/i2t``):

    captions.jsonl        one row per image, in input order: image, caption,
                          n_tokens, n_nontext, finish_reason
    captions_NNN.jsonl    the same rows per GPU shard

Every visible GPU is used.

Usage:
    python caption_images.py --images examples/sample.jpg
    python caption_images.py --images /path/to/image_dir more.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

import vocab
from common import (
    DEFAULT_MODEL,
    TOKENIZERS_REPO,
    launch_workers,
    resolve,
    resolve_text_tokenizer,
    silence_tokenizerless_warmup,
)
from vllm import LLM, SamplingParams, TokensPrompt

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_PRINTED = 20


def check_checkpoint(model_dir: str | Path) -> None:
    """Refuse a checkpoint whose vocabulary is not the Halcyon layout."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return
    vocab_size = json.loads(config_path.read_text()).get("vocab_size")
    if vocab_size is not None and vocab_size != vocab.TOTAL_VOCAB_SIZE:
        raise RuntimeError(
            f"{model_dir} has vocab_size {vocab_size}; Halcyon checkpoints have "
            f"{vocab.TOTAL_VOCAB_SIZE}"
        )


def list_images(paths: list[str]) -> list[Path]:
    """Expand files and directories (non-recursive, sorted) into image paths."""
    images: list[Path] = []
    for item in map(Path, paths):
        if item.is_dir():
            images.extend(
                sorted(p for p in item.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif item.is_file():
            images.append(item)
        else:
            raise FileNotFoundError(item)
    if not images:
        raise ValueError(f"no images found in {paths}")
    return images


def encode_images(
    paths: list[Path], image_tokenizer: str, batch_size: int
) -> list[list[int]]:
    """Encode image files to ``[SOV] + image token ids + [EOV]`` prompts.

    The image tokenizer is freed afterwards so vLLM can claim the GPU memory.
    """
    import torch
    from PIL import Image

    from image_tokenizer import HF_WEIGHTS, ImageTokenizer

    tok = ImageTokenizer.from_pretrained(
        resolve(image_tokenizer, HF_WEIGHTS), device="cuda", dtype=torch.bfloat16
    )
    prefixes: list[list[int]] = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                batch.append(image.convert("RGB"))
        codes = tok.encode(tok.preprocess(batch)).reshape(len(batch), -1)
        for row in (codes + vocab.IMG_OFFSET).cpu().tolist():
            prefixes.append([vocab.SOV_ID, *row, vocab.EOV_ID])
    del tok
    torch.cuda.empty_cache()
    return prefixes


def sampling_params(args: argparse.Namespace, index: int) -> SamplingParams:
    """Per-image params; the seed depends only on the image's input index."""
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else -1,
        top_p=args.top_p if 0.0 < args.top_p < 1.0 else 1.0,
        stop_token_ids=[vocab.EOS_ID],
        detokenize=False,
        seed=args.seed + index,
    )


def run(args: argparse.Namespace) -> None:
    args.model = resolve(args.model, revision=args.revision)
    check_checkpoint(args.model)
    image_paths = list_images(args.images)
    count = len(image_paths) if args.limit == 0 else min(args.limit, len(image_paths))
    indices = list(range(args.shard_index, count, args.num_shards))
    print(f"shard {args.shard_index}/{args.num_shards}: {len(indices)} of {count} images")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"captions_{args.shard_index:03d}.jsonl"
    if not indices:
        output_path.write_text("")
        return

    prefixes = encode_images(
        [image_paths[i] for i in indices], args.image_tokenizer, args.encode_batch_size
    )
    engine = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=1280,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=prefix) for prefix in prefixes],
        [sampling_params(args, i) for i in indices],
    )

    tokenizer = Tokenizer.from_file(resolve_text_tokenizer(args.text_tokenizer))
    with output_path.open("w", encoding="utf-8") as handle:
        for index, output in zip(indices, outputs, strict=True):
            ids = [t for t in output.outputs[0].token_ids if t != vocab.EOS_ID]
            text_ids = [t for t in ids if t < vocab.TEXT_VOCAB_SIZE]
            record = {
                "index": index,
                "image": str(image_paths[index]),
                "caption": tokenizer.decode(text_ids),
                "n_tokens": len(ids),
                "n_nontext": len(ids) - len(text_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(indices)} captions to {output_path}")


def merge_shards(out_dir: Path, num_shards: int) -> Path:
    """Concatenate ``captions_NNN.jsonl`` into ``captions.jsonl`` by input order."""
    records = []
    for shard in range(num_shards):
        with (out_dir / f"captions_{shard:03d}.jsonl").open(encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    records.sort(key=lambda record: record["index"])
    merged = out_dir / "captions.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    for record in records[:MAX_PRINTED]:
        print(f"{record['image']}: {record['caption']}")
    if len(records) > MAX_PRINTED:
        print(f"... {len(records) - MAX_PRINTED} more")
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="image files and/or directories, any resolution",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"hub repo id or local model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="hub branch; 'main' (I2T 10%%) is the captioning model",
    )
    parser.add_argument("--out-dir", default="outputs/i2t")
    parser.add_argument(
        "--text-tokenizer",
        default=TOKENIZERS_REPO,
        help="tokenizer.json, or the hub repo holding it",
    )
    parser.add_argument(
        "--image-tokenizer",
        default=TOKENIZERS_REPO,
        help="DI8x8 weights (.pt), or the hub repo holding image_tokenizer/",
    )
    parser.add_argument("--max-tokens", type=int, default=254)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    parser.add_argument("--top-p", type=float, default=0.0, help="0 disables top-p")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0, help="only the first N images")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="GPUs to use, one worker each; 0 uses every visible GPU",
    )
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.limit < 0 or args.num_gpus < 0 or args.encode_batch_size < 1:
        parser.error("--limit/--num-gpus must be >= 0, --encode-batch-size >= 1")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    workers = launch_workers(args.num_gpus, args.num_shards, out_dir)
    if not workers:
        silence_tokenizerless_warmup()
        run(args)
        if args.num_shards > 1:
            return  # an external launcher handles the merge
    print(f"captions in {merge_shards(out_dir, max(workers, 1))}")


if __name__ == "__main__":
    main()
