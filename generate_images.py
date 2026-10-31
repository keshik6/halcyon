"""Text-to-image sampling for Halcyon checkpoints with vLLM.

Each sample is ``{caption} [SOV]`` followed by exactly 1,024 image tokens,
sampled with classifier-free guidance with the ``[SOV]`` prefix (the
caption-dropout rows seen in training). The conditional and unconditional
branches are two vLLM requests combined by :class:`cfg.CFGLogitsProcessor`.

Outputs under ``--out-dir`` (default ``outputs/t2i``):

    images/{row:06d}_{sample}.png   256x256 decoded images
    codes_NNN.bin                   uint16 FSQ codes, 1,024 per sample, per GPU shard
    manifest_NNN.jsonl              row, sample, key, caption and seed per sample

Row is the caption's index in the input and the seed is ``--seed`` + global
sample id, so outputs do not depend on the GPU count. Every visible GPU is used.
Runs resume from the last completed wave.

Usage:
    python generate_images.py --caption "A red boat on a calm lake at sunrise."
    python generate_images.py --prompts examples/prompts.jsonl
    python generate_images.py --model radicalnumerics/Halcyon-0.52B --prompts p.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np
from tokenizers import Tokenizer

import vocab
from cfg import COND, UNCOND, CFGLogitsProcessor, cfg_extra_args
from common import (
    DEFAULT_MODEL,
    TOKENIZERS_REPO,
    default_cfg_scale,
    launch_workers,
    resolve,
    resolve_text_tokenizer,
    silence_tokenizerless_warmup,
)
from vllm import LLM, SamplingParams, TokensPrompt, envs

CODES_DTYPE = np.uint16


def is_guided(cfg_scale: float) -> bool:
    """Guidance at scale 1.0 is the conditional distribution, so it is skipped."""
    return cfg_scale != 1.0


def seqs_per_pair(guided: bool) -> int:
    """Requests submitted per sample: the caption branch, plus its partner."""
    return 2 if guided else 1


def load_prompts(path: str | Path, field: str = "caption") -> list[dict[str, Any]]:
    """Read captions from a JSONL file (``field`` per row) or a plain text file."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if field not in record:
                    raise KeyError(f"{path}: row is missing the {field!r} field")
                rows.append({"caption": record[field], "key": record.get("key")})
        else:
            for line in handle:
                caption = line.strip()
                if caption:
                    rows.append({"caption": caption, "key": None})
    if not rows:
        raise ValueError(f"no prompts found in {path}")
    return rows


def _encode(captions: list[str], tokenizer_json: str, max_tokens: int) -> list[list[int]]:
    """Tokenize stripped captions as in training, truncated to the context window."""
    tokenizer = Tokenizer.from_file(tokenizer_json)
    encoded = tokenizer.encode_batch([caption.strip() for caption in captions])
    return [enc.ids[:max_tokens] for enc in encoded]


def build_requests(
    caption_ids: list[list[int]],
    row_indices: list[int],
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    seed_base: int,
    max_tokens: int = vocab.IMAGE_TOKENS,
) -> tuple[list[Any], list[Any], list[int]]:
    """Build one (or two, under guidance) vLLM requests per sample.

    Returns the prompts, their sampling params, and the positions in the request
    list holding each sample's conditional branch - the only branch whose output
    is kept.
    """
    guided = is_guided(cfg_scale)
    prompts: list[Any] = []
    params: list[Any] = []
    cond_positions: list[int] = []

    for row, ids in zip(row_indices, caption_ids, strict=True):
        seed = seed_base + row
        common = {
            "temperature": temperature,
            "top_k": max(0, top_k),
            "top_p": top_p if 0.0 < top_p < 1.0 else 1.0,
            "max_tokens": max_tokens,
            # Nothing can stop generation early: the codebook span excludes every
            # special id, so [EOV] and EOS are unreachable and the token budget
            # is what ends the sequence.
            "ignore_eos": True,
            "detokenize": False,
            "seed": seed,
        }
        cond_positions.append(len(prompts))
        prompts.append(TokensPrompt(prompt_token_ids=[*ids, vocab.SOV_ID]))
        params.append(
            SamplingParams(
                **common,
                extra_args=cfg_extra_args(
                    group=row,
                    role=COND,
                    scale=cfg_scale,
                    token_range=vocab.IMAGE_TOKEN_RANGE,
                ),
            )
        )
        if guided:
            # The unconditional branch drops the caption but keeps the structure,
            # matching the caption-dropout documents the checkpoint saw in training.
            prompts.append(TokensPrompt(prompt_token_ids=[vocab.SOV_ID]))
            params.append(
                SamplingParams(
                    **common,
                    extra_args=cfg_extra_args(
                        group=row,
                        role=UNCOND,
                        scale=cfg_scale,
                        token_range=vocab.IMAGE_TOKEN_RANGE,
                    ),
                )
            )

    return prompts, params, cond_positions


def build_engine(
    model: str,
    pairs_per_wave: int,
    guided: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    seed: int,
    max_caption_tokens: int,
) -> Any:
    """Construct the offline engine with the scheduling guarantees CFG needs."""
    # Both branches of a pair must enter the running batch in the same scheduler
    # step. With the engine core in its own process, LLM.generate ships requests
    # over ZMQ one at a time while the core's busy loop is already running, so it
    # can schedule a conditional branch a step before its partner arrives and the
    # pair is desynchronised before decoding starts. Running the core in-process
    # makes add_request enqueue directly, so a whole wave is waiting before the
    # first step. Must be set before the engine is constructed.
    # Read lazily by vllm.envs, and the engine only caches environment variables
    # once EngineCore starts, so setting it here still takes effect.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Guidance relies on both branches drawing the same sampling noise from a
    # per-request torch.Generator, which is the V1 model runner's scheme. The V2
    # runner instead derives Gumbel noise from (seed, absolute token position);
    # the two branches sit at different positions, because only one of them
    # carries the caption, so they would draw different noise and diverge on the
    # first token. Requesting a custom logits processor already makes vLLM fall
    # back to V1, but an explicit VLLM_USE_V2_MODEL_RUNNER=1 would override that.
    if envs.VLLM_USE_V2_MODEL_RUNNER:
        raise RuntimeError(
            "VLLM_USE_V2_MODEL_RUNNER=1 is incompatible with classifier-free "
            "guidance: the V2 sampler seeds its noise by absolute token position, "
            "so the conditional and unconditional branches cannot stay in lockstep"
        )

    seqs_per_wave = seqs_per_pair(guided) * pairs_per_wave
    # Admitting a whole wave in one scheduler step keeps a pair's two branches
    # from being split across steps, which would desynchronise their samplers.
    # Enough to prefill a whole wave in one step, but never below max_model_len:
    # with chunked prefill off, SchedulerConfig.verify_max_model_len rejects a
    # budget under that. The other enforced bound, budget >= max_num_seqs, holds
    # already because every sequence contributes at least one prompt token.
    prefill_budget = max(seqs_per_wave * (max_caption_tokens + 2), max_model_len)
    return LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=seqs_per_wave,
        max_num_batched_tokens=prefill_budget,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
        logits_processors=[CFGLogitsProcessor],
        seed=seed,
    )


def check_wave_fits_kv_cache(llm: Any, pairs_per_wave: int, guided: bool) -> None:
    """Fail now if a wave cannot be held in KV cache to full length.

    Preemption is what breaks guidance: a branch rescheduled apart from its
    partner falls out of sampler lockstep. The logits processor detects that and
    raises, but only once decoding is under way, so the capacity is checked here
    instead of hours into a run.
    """
    cache_config = llm.llm_engine.vllm_config.cache_config
    # The concurrency the engine computed at max_model_len tokens per request,
    # which is the length every request here reaches. Preferred over
    # num_gpu_blocks * block_size, which vllm/config/cache.py documents as wrong
    # for models whose requests span several KV cache groups.
    affordable = int(cache_config.kv_cache_max_concurrency)
    per_pair = seqs_per_pair(guided)
    needed = per_pair * pairs_per_wave
    if needed > affordable:
        raise RuntimeError(
            f"KV cache holds {affordable} concurrent full-length sequences but a "
            f"wave of {pairs_per_wave} samples needs {needed}; requests would be "
            "preempted and the guidance branches would desynchronise. Lower "
            f"--pairs-per-wave to {max(affordable // per_pair, 1)} or raise "
            "--gpu-memory-utilization."
        )


def _resume_state(codes_path: Path, manifest_path: Path) -> int:
    """Return the number of complete rows already on disk, truncating any tail.

    A killed job can leave the codes file longer than the manifest (the manifest
    is written second); trimming to the manifest keeps the two in step.
    """
    row_bytes = vocab.IMAGE_TOKENS * np.dtype(CODES_DTYPE).itemsize
    if not manifest_path.exists():
        if codes_path.exists() and codes_path.stat().st_size:
            raise RuntimeError(
                f"{codes_path} exists without {manifest_path}; cannot determine a "
                "safe resume point. Pass --no-resume to replace both outputs."
            )
        return 0
    with manifest_path.open() as handle:
        done = sum(1 for line in handle if line.strip())
    expected = done * row_bytes
    actual = codes_path.stat().st_size if codes_path.exists() else 0
    if actual < expected:
        raise RuntimeError(
            f"{manifest_path} records {done} samples ({expected} code bytes), but "
            f"{codes_path} has only {actual} bytes; cannot resume without losing "
            "manifest-to-code alignment. Pass --no-resume to replace both outputs."
        )
    if actual > expected:
        with codes_path.open("r+b") as handle:
            handle.truncate(expected)
    return done


@dataclass(frozen=True)
class ShardPlan:
    """The samples one shard owns, and their identity in the whole prompt set."""

    start: int
    """Index of this shard's first prompt within the full prompt file."""
    rows: list[dict[str, Any]]
    """Prompts assigned to this shard."""
    jobs: list[tuple[int, int]]
    """One (prompt index within the shard, draw) pair per sample."""
    ids: list[int]
    """Global sample id per job, used as the guidance group and the seed offset."""


def plan_shard(
    rows: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
    samples_per_prompt: int,
) -> ShardPlan:
    """Split the prompt set into contiguous shards and enumerate their samples.

    Shards are contiguous so each one's output is independently resumable, and
    sample ids are global so a sample's identity, and therefore its seed, does
    not depend on how the set was split.
    """
    per_shard = -(-len(rows) // num_shards)
    start = shard_index * per_shard
    shard_rows = rows[start : start + per_shard]
    jobs = [
        (local_row, draw)
        for local_row in range(len(shard_rows))
        for draw in range(samples_per_prompt)
    ]
    ids = [(start + local_row) * samples_per_prompt + draw for local_row, draw in jobs]
    return ShardPlan(start=start, rows=shard_rows, jobs=jobs, ids=ids)


def codes_from_outputs(
    outputs: list[Any], cond_positions: list[int], sample_ids: list[int]
) -> np.ndarray:
    """Extract each sample's conditional branch as codebook indices.

    The unconditional branch generates the same tokens by construction, so only
    the conditional one is kept.
    """
    codes = np.empty((len(cond_positions), vocab.IMAGE_TOKENS), dtype=CODES_DTYPE)
    for slot, position in enumerate(cond_positions):
        token_ids = outputs[position].outputs[0].token_ids
        if len(token_ids) != vocab.IMAGE_TOKENS:
            raise RuntimeError(
                f"sample {sample_ids[slot]} produced {len(token_ids)} tokens, "
                f"expected {vocab.IMAGE_TOKENS}"
            )
        codes[slot] = np.asarray(token_ids, dtype=np.int64) - vocab.IMG_OFFSET
    return codes


def append_wave(
    codes_file: BinaryIO,
    manifest_file: TextIO,
    codes: np.ndarray,
    plan: ShardPlan,
    wave: list[int],
    seed_base: int,
) -> None:
    """Append one wave's codes and manifest rows, codes first.

    The order is what makes a killed job resumable: writing codes before the
    manifest means an interrupted wave leaves the codes file long rather than
    short, which :func:`_resume_state` trims back to the manifest.
    """
    codes.tofile(codes_file)
    codes_file.flush()
    for job in wave:
        local_row, draw = plan.jobs[job]
        manifest_file.write(
            json.dumps(
                {
                    "row": plan.start + local_row,
                    "sample": draw,
                    "key": plan.rows[local_row]["key"],
                    "caption": plan.rows[local_row]["caption"],
                    "seed": seed_base + plan.ids[job],
                }
            )
            + "\n"
        )
    manifest_file.flush()


def save_pngs(
    image_tokenizer: Any,
    codes: np.ndarray,
    plan: ShardPlan,
    wave: list[int],
    png_dir: Path,
    batch_size: int,
) -> None:
    """Decode one wave's codes and write ``{row:06d}_{sample}.png`` per sample.

    Called before :func:`append_wave`, so a sample is only recorded as done once
    its PNG exists; a resumed job regenerates (same seed) and overwrites any
    PNGs from an interrupted wave.
    """
    for start in range(0, len(wave), batch_size):
        images = image_tokenizer.to_pil(
            image_tokenizer.decode(codes[start : start + batch_size].astype(np.int64))
        )
        for job, image in zip(wave[start : start + batch_size], images, strict=True):
            local_row, draw = plan.jobs[job]
            image.save(png_dir / f"{plan.start + local_row:06d}_{draw}.png")


def check_vocab_layout(model_dir: str) -> None:
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


def run(args: argparse.Namespace) -> None:
    if args.cfg_scale is None:
        args.cfg_scale = default_cfg_scale(args.model, args.revision)
    args.model = resolve(args.model, revision=args.revision)
    check_vocab_layout(args.model)
    if args.caption:
        rows = [{"caption": caption, "key": None} for caption in args.caption]
    else:
        rows = load_prompts(args.prompts, args.caption_field)
    if args.limit:
        rows = rows[: args.limit]

    plan = plan_shard(rows, args.num_shards, args.shard_index, args.samples_per_prompt)
    if not plan.rows:
        print(f"shard {args.shard_index}/{args.num_shards} is empty; nothing to do")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_path = out_dir / f"codes_{args.shard_index:03d}.bin"
    manifest_path = out_dir / f"manifest_{args.shard_index:03d}.jsonl"

    done = _resume_state(codes_path, manifest_path) if args.resume else 0
    if done:
        with manifest_path.open() as handle:
            written = [json.loads(line)["caption"] for line in handle if line.strip()]
        expected = [
            plan.rows[plan.jobs[j][0]]["caption"]
            for j in range(min(done, len(plan.jobs)))
        ]
        if written != expected:
            raise SystemExit(
                f"{out_dir} holds samples for different captions; pass a new "
                "--out-dir, or --no-resume to overwrite"
            )
    if done >= len(plan.jobs):
        print(f"shard {args.shard_index}: all {done} samples already written")
        return
    if done:
        print(f"shard {args.shard_index}: resuming after {done} samples")

    max_caption = vocab.max_caption_tokens(args.max_model_len)
    caption_ids = _encode(
        [row["caption"] for row in plan.rows],
        resolve_text_tokenizer(args.text_tokenizer),
        max_caption,
    )

    image_tokenizer = None
    png_dir = out_dir / "images"
    if args.png:
        from image_tokenizer import HF_WEIGHTS, ImageTokenizer

        png_dir.mkdir(exist_ok=True)
        image_tokenizer = ImageTokenizer.from_pretrained(
            resolve(args.image_tokenizer, HF_WEIGHTS), device="cuda"
        )

    guided = is_guided(args.cfg_scale)
    llm = build_engine(
        model=args.model,
        pairs_per_wave=args.pairs_per_wave,
        guided=guided,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_caption_tokens=max_caption,
    )
    check_wave_fits_kv_cache(llm, pairs_per_wave=args.pairs_per_wave, guided=guided)

    pending = list(range(done, len(plan.jobs)))
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(pending)} samples over "
        f"{len(plan.rows)} prompts, cfg {args.cfg_scale}, temp {args.temperature}, "
        f"top_k {args.top_k}, top_p {args.top_p}, {args.pairs_per_wave} pairs/wave"
    )

    started = time.time()
    completed = 0
    codes_mode = "ab" if args.resume else "wb"
    manifest_mode = "a" if args.resume else "w"
    with (
        codes_path.open(codes_mode) as codes_file,
        manifest_path.open(manifest_mode) as manifest_file,
    ):
        for offset in range(0, len(pending), args.pairs_per_wave):
            wave = pending[offset : offset + args.pairs_per_wave]
            sample_ids = [plan.ids[j] for j in wave]
            prompts, params, cond_positions = build_requests(
                caption_ids=[caption_ids[plan.jobs[j][0]] for j in wave],
                row_indices=sample_ids,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed_base=args.seed,
            )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            codes = codes_from_outputs(outputs, cond_positions, sample_ids)
            if image_tokenizer is not None:
                save_pngs(
                    image_tokenizer, codes, plan, wave, png_dir, args.decode_batch_size
                )
            append_wave(
                codes_file,
                manifest_file,
                codes=codes,
                plan=plan,
                wave=wave,
                seed_base=args.seed,
            )

            completed += len(wave)
            elapsed = time.time() - started
            rate = completed / elapsed
            remaining = (len(pending) - completed) / rate if rate else float("nan")
            print(
                f"  {completed}/{len(pending)} samples "
                f"({rate:.1f} img/s, {rate * vocab.IMAGE_TOKENS:.0f} tok/s, "
                f"eta {remaining / 60:.1f} min)",
                flush=True,
            )

    print(f"shard {args.shard_index}: wrote {completed} samples to {codes_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"hub repo id or local model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--revision",
        default="baseline",
        help="hub branch: 'baseline' (paper T2I model, default) or 'main' (I2T 10%%)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompts",
        help="JSONL with a caption field, or one caption per line",
    )
    source.add_argument(
        "--caption",
        action="append",
        help="a caption to generate from; repeat for several",
    )
    parser.add_argument("--out-dir", default="outputs/t2i")
    parser.add_argument("--caption-field", default="caption")
    parser.add_argument(
        "--text-tokenizer",
        default=TOKENIZERS_REPO,
        help="tokenizer.json, or the hub repo holding it",
    )
    parser.add_argument(
        "--image-tokenizer",
        default=TOKENIZERS_REPO,
        help="DI8x8 weights (.pt) or the hub repo holding image_tokenizer/",
    )
    parser.add_argument(
        "--no-png",
        dest="png",
        action="store_false",
        help="write codes and manifest only; skip decoding to PNG",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=16,
        help="images per image-tokenizer decode call",
    )

    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="independent draws per caption; each gets its own seed",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="guidance scale; default is the paper's per-model value (Table 16)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    parser.add_argument("--top-p", type=float, default=0.0, help="0 disables top-p")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument(
        "--pairs-per-wave",
        type=int,
        default=128,
        help="samples in flight; KV cache must hold 2x this many full-length sequences",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="GPUs to use, one worker each; 0 uses every visible GPU",
    )
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--limit", type=int, default=0, help="only sample the first N prompts"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore existing output and start over (default: resume)",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.samples_per_prompt < 1:
        parser.error("--samples-per-prompt must be positive")
    if args.pairs_per_wave < 1:
        parser.error("--pairs-per-wave must be positive")
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be positive")
    if args.max_model_len <= vocab.IMAGE_TOKENS + 1:
        parser.error("--max-model-len must leave room for caption tokens")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.num_gpus < 0:
        parser.error("--num-gpus must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    if launch_workers(args.num_gpus, args.num_shards, Path(args.out_dir)):
        print(f"all shards done; images in {Path(args.out_dir) / 'images'}")
        return
    # Sharded runs each own one GPU; keep vLLM from probing the whole node.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    silence_tokenizerless_warmup()
    run(args)


if __name__ == "__main__":
    main()
