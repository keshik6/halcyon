"""Shared helpers: HF paths, paper defaults, and a multi-GPU launcher."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "radicalnumerics/Halcyon-1.6B"
TOKENIZERS_REPO = "radicalnumerics/Halcyon-Tokenizers"
WORKER_ENV = "HALCYON_WORKER"

# Paper Table 16: CFG scale with the lowest gFD on GPIC Val-50K, per model and
# branch ("baseline": caption dropout 0.1; "main": I2T 10%, no caption dropout).
PAPER_CFG = {
    "Halcyon-0.1B": {"baseline": 7.5, "main": 7.5},
    "Halcyon-0.25B": {"baseline": 6.0, "main": 6.0},
    "Halcyon-0.52B": {"baseline": 6.0, "main": 6.0},
    "Halcyon-1.6B": {"baseline": 5.0, "main": 5.2},
    "Halcyon-4B": {"baseline": 4.5, "main": 4.6},
}


def default_cfg_scale(model: str, revision: str | None) -> float:
    """The paper's CFG scale for a released model, or an error asking for one."""
    scales = PAPER_CFG.get(Path(model.rstrip("/")).name, {})
    if (revision or "main") in scales:
        return scales[revision or "main"]
    raise SystemExit(f"no paper CFG scale for {model}@{revision}; pass --cfg-scale")


def resolve(
    path_or_repo: str, filename: str | None = None, revision: str | None = None
) -> str:
    """Return a local path, downloading from the Hugging Face Hub if needed.

    With ``filename`` a single file is fetched from the repo; otherwise the whole
    repo snapshot is, at ``revision`` (branch, tag or commit; default ``main``).
    """
    if Path(path_or_repo).exists():
        return path_or_repo
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename:
        return hf_hub_download(path_or_repo, filename, revision=revision)
    return snapshot_download(path_or_repo, revision=revision)


def resolve_text_tokenizer(tokenizer: str) -> str:
    """Local path of the text ``tokenizer.json``, given the file or its hub repo."""
    if tokenizer.endswith(".json"):
        return tokenizer
    return resolve(tokenizer, "tokenizer.json")


class _TokenizerlessWarmupFilter(logging.Filter):
    """Drop vLLM's chat-template warmup traceback.

    ``LLM.__init__`` always warms up the chat template, which needs a tokenizer.
    These scripts run with ``skip_tokenizer_init=True`` and tokenize themselves,
    so the warmup can only fail, and nothing here uses a chat template.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("Chat template warmup failed")


def silence_tokenizerless_warmup() -> None:
    """Install :class:`_TokenizerlessWarmupFilter` on vLLM's renderer logger."""
    logging.getLogger("vllm.renderers.base").addFilter(_TokenizerlessWarmupFilter())


def visible_gpus() -> list[str]:
    """GPU ids this process may use, from ``CUDA_VISIBLE_DEVICES`` or the driver."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [d for d in visible.split(",") if d.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def launch_workers(num_gpus: int, num_shards: int, out_dir: Path) -> int:
    """Re-run this script once per GPU as shard workers; return the worker count.

    Does nothing (returns 0) inside a worker, when sharding was given
    explicitly (``--num-shards`` > 1, e.g. from the sbatch launchers), or when
    only one GPU is in use. ``num_gpus`` 0 means every visible GPU. Each worker
    logs to ``{out_dir}/shard_{i}.log``.
    """
    if os.environ.get(WORKER_ENV) or num_shards > 1:
        return 0
    devices = visible_gpus()
    if num_gpus:
        if num_gpus > len(devices):
            raise SystemExit(f"--num-gpus {num_gpus}, but {len(devices)} GPUs visible")
        devices = devices[:num_gpus]
    if len(devices) <= 1:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for shard, device in enumerate(devices):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": device, WORKER_ENV: "1"}
        log = (out_dir / f"shard_{shard}.log").open("w")
        argv = [
            *sys.argv,
            "--num-shards",
            str(len(devices)),
            "--shard-index",
            str(shard),
        ]
        workers.append(
            (
                subprocess.Popen(
                    [sys.executable, *argv],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
            )
        )
        print(f"shard {shard} -> GPU {device}, log {log.name}", flush=True)

    failed = []
    try:
        for shard, (proc, log) in enumerate(workers):
            if proc.wait():
                failed.append(shard)
            log.close()
    except KeyboardInterrupt:
        for proc, _ in workers:
            proc.terminate()
        raise
    if failed:
        for shard in failed:
            tail = (out_dir / f"shard_{shard}.log").read_text().splitlines()[-20:]
            print(f"--- shard {shard} failed; last lines of its log:", *tail, sep="\n")
        raise SystemExit(1)
    return len(devices)
