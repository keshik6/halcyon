"""Classifier-free guidance for vLLM, expressed as a batch logits processor.

Classifier-free guidance needs two forward passes per generated token - one on
the conditional prompt, one on an unconditional prompt - combined as

    guided = uncond + scale * (cond - uncond)

Per-request configuration passed in ``SamplingParams.extra_args``; build it with
:func:`cfg_extra_args`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

EXTRA_ARGS_KEY = "cfg"
COND = "cond"
UNCOND = "uncond"


def cfg_extra_args(
    group: str | int,
    role: str,
    scale: float,
    token_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the ``extra_args`` payload that opts a request into guidance.

    Args:
        group: Identifier shared by the two requests forming one pair.
        role: ``"cond"`` or ``"uncond"``.
        scale: Guidance scale. 1.0 reproduces the conditional distribution.
        token_range: Optional ``[lo, hi)`` span of token ids the request may
            emit. Everything outside is driven to ``-inf`` *after* the branches
            are combined. Restricting the vocabulary here rather than through
            ``SamplingParams.allowed_token_ids`` is deliberate: that whitelist is
            applied before logits processors run, so both branches would arrive
            holding ``-inf`` on the excluded ids and the combine would evaluate
            ``-inf + scale * nan``.
    """
    if role not in (COND, UNCOND):
        raise ValueError(f"role must be {COND!r} or {UNCOND!r}, got {role!r}")
    payload: dict[str, Any] = {"group": group, "role": role, "scale": float(scale)}
    if token_range is not None:
        lo, hi = token_range
        if not 0 <= lo < hi:
            raise ValueError(
                f"token_range must satisfy 0 <= lo < hi, got {token_range}"
            )
        payload["token_range"] = [int(lo), int(hi)]
    return {EXTRA_ARGS_KEY: payload}


@dataclass
class _Branch:
    """One side of a guidance pair, as tracked at a persistent-batch index."""

    group: Any
    role: str
    scale: float
    token_range: tuple[int, int] | None
    # Live reference to the request's generated tokens, per the BatchUpdate
    # contract; used to verify the two branches have not drifted apart.
    output_tok_ids: list[int]


def _parse(params: SamplingParams) -> dict[str, Any] | None:
    extra = params.extra_args
    if not extra or EXTRA_ARGS_KEY not in extra:
        return None
    payload = extra[EXTRA_ARGS_KEY]
    if not isinstance(payload, dict):
        raise TypeError(f"extra_args[{EXTRA_ARGS_KEY!r}] must be a dict")
    if payload.get("role") not in (COND, UNCOND):
        raise ValueError(
            f"extra_args[{EXTRA_ARGS_KEY!r}]['role'] must be {COND!r} or {UNCOND!r}"
        )
    if "group" not in payload:
        raise ValueError(f"extra_args[{EXTRA_ARGS_KEY!r}] requires a 'group'")
    return payload


class CFGLogitsProcessor(LogitsProcessor):
    """Combine paired conditional/unconditional requests into a guided one.

    Register with ``LLM(..., logits_processors=[CFGLogitsProcessor])``. Requests
    that carry no CFG payload are left untouched.

    Offline use only. The OpenAI server passes custom arguments through
    ``vllm_xargs``, whose values are typed ``str | int | float``, so it cannot
    carry the nested payload :func:`cfg_extra_args` builds.
    """

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.is_pin_memory = is_pin_memory
        self._branches: dict[int, _Branch] = {}
        self._dirty = True
        # Rebuilt from _branches whenever the batch changes.
        self._cond_idx: torch.Tensor | None = None
        self._uncond_idx: torch.Tensor | None = None
        self._pair_idx: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._span: tuple[int, int] | None = None
        self._cond_rows: list[int] = []
        self._uncond_rows: list[int] = []
        self._unpaired: list[int] = []
        self._warned_unpaired = False

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        payload = _parse(sampling_params)
        if payload is None:
            return
        if sampling_params.seed is None:
            raise ValueError(
                "guided requests need an explicit SamplingParams.seed: both "
                "branches must draw the same sampling noise to stay in lockstep"
            )
        if "token_range" in payload:
            lo, hi = payload["token_range"]
            if not 0 <= lo < hi:
                raise ValueError(f"invalid cfg token_range {(lo, hi)}")
        # Penalties are applied after this processor, from each request's own
        # prompt token ids. The two branches carry different prompts, so a
        # nonzero penalty gives them different logits and breaks the lockstep
        # that the shared seed depends on.
        penalties = {
            "repetition_penalty": sampling_params.repetition_penalty - 1.0,
            "frequency_penalty": sampling_params.frequency_penalty,
            "presence_penalty": sampling_params.presence_penalty,
        }
        offenders = sorted(name for name, value in penalties.items() if value)
        if offenders:
            raise ValueError(
                f"guided requests cannot use {', '.join(offenders)}: penalties are "
                "applied per request from its own prompt, which would desynchronise "
                "the two branches"
            )

    def is_argmax_invariant(self) -> bool:
        """Guidance reweights the distribution, so it can move the argmax."""
        return False

    @staticmethod
    def _new_branch(
        params: SamplingParams,
        _prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> _Branch | None:
        payload = _parse(params)
        if payload is None:
            return None
        token_range = payload.get("token_range")
        return _Branch(
            group=payload["group"],
            role=payload["role"],
            scale=float(payload.get("scale", 1.0)),
            token_range=tuple(token_range) if token_range else None,
            output_tok_ids=output_tok_ids,
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if process_dict_updates(self._branches, batch_update, self._new_branch):
            self._dirty = True

    def _rebuild(self) -> None:
        self._dirty = False
        cond: dict[Any, int] = {}
        uncond: dict[Any, int] = {}
        for index, branch in self._branches.items():
            side = cond if branch.role == COND else uncond
            if branch.group in side:
                raise ValueError(
                    f"two {branch.role!r} requests share cfg group {branch.group!r}; "
                    "group ids must be unique per pair"
                )
            side[branch.group] = index

        spans = {b.token_range for b in self._branches.values()}
        if len(spans) > 1:
            raise ValueError(
                "all guided requests in a batch must share one token_range, "
                f"got {spans}"
            )
        self._span = next(iter(spans)) if spans else None

        paired = cond.keys() & uncond.keys()
        groups = sorted(paired, key=repr)
        self._cond_rows = [cond[g] for g in groups]
        self._uncond_rows = [uncond[g] for g in groups]
        self._unpaired = [
            idx
            for group, idx in (*cond.items(), *uncond.items())
            if group not in paired
        ]

        if groups:
            self._cond_idx = torch.tensor(
                self._cond_rows, dtype=torch.long, device=self.device
            )
            self._uncond_idx = torch.tensor(
                self._uncond_rows, dtype=torch.long, device=self.device
            )
            self._scale = torch.tensor(
                [self._branches[i].scale for i in self._cond_rows],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)
        else:
            self._cond_idx = self._uncond_idx = self._scale = None

        guided_rows = [*self._cond_rows, *self._uncond_rows, *self._unpaired]
        self._pair_idx = (
            torch.tensor(guided_rows, dtype=torch.long, device=self.device)
            if guided_rows
            else None
        )

        if self._unpaired and not self._warned_unpaired:
            self._warned_unpaired = True
            logger.warning(
                "%d guided request(s) have no partner in the batch and will "
                "sample unguided; their partner was never submitted, finished "
                "early, or was aborted.",
                len(self._unpaired),
            )

    def _check_alignment(self) -> None:
        """Fail loudly if a pair's two branches have drifted apart.

        Both branches receive the same logits and the same sampling seed. 
        """
        for cond_row, uncond_row in zip(
            self._cond_rows, self._uncond_rows, strict=True
        ):
            cond_out = self._branches[cond_row].output_tok_ids
            uncond_out = self._branches[uncond_row].output_tok_ids
            if len(cond_out) != len(uncond_out) or (
                cond_out and cond_out[-1] != uncond_out[-1]
            ):
                raise RuntimeError(
                    "classifier-free guidance branches are out of step for group "
                    f"{self._branches[cond_row].group!r}: conditional has "
                    f"{len(cond_out)} tokens, unconditional has {len(uncond_out)}. "
                    "The two requests were not scheduled together. Submit pairs in "
                    "waves that fit max_num_seqs, leave enough KV cache that no "
                    "request is preempted, and run the engine core in-process "
                    "(VLLM_ENABLE_V1_MULTIPROCESSING=0) so a whole wave is queued "
                    "before the first scheduler step."
                )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._branches:
            return logits
        if self._dirty:
            self._rebuild()

        # Restrict the guided requests to their permitted span. Done before the
        # combine so the excluded ids are already unreachable, and without
        # touching the span itself so the combine below sees real logits.
        if self._span is not None and self._pair_idx is not None:
            lo, hi = self._span
            if lo > 0:
                logits[self._pair_idx, :lo] = float("-inf")
            if hi < logits.shape[1]:
                logits[self._pair_idx, hi:] = float("-inf")

        if self._cond_idx is None:
            return logits

        self._check_alignment()

        lo, hi = self._span if self._span is not None else (0, logits.shape[1])
        cond = logits[self._cond_idx, lo:hi]
        uncond = logits[self._uncond_idx, lo:hi]
        guided = torch.lerp(uncond, cond, self._scale)
        logits[self._cond_idx, lo:hi] = guided
        logits[self._uncond_idx, lo:hi] = guided
        return logits
