"""Standalone DI8x8 image tokenizer: 256x256 RGB <-> 32x32 FSQ codes.

Module and buffer names match the training code exactly, so the released
``di8x8_242172_ema.pt`` state dict loads with ``strict=True``; any architectural
drift will fail at load.

Codes are FSQ codebook indices in ``[0, 64000)``, raster order, i.e. the LM's
image token id minus ``vocab.IMG_OFFSET``.

    tok = ImageTokenizer.from_pretrained(device="cuda")
    codes = tok.encode(tok.preprocess([pil]))         # (B, 32, 32) int64
    pils = tok.to_pil(tok.decode(codes))              # list of 256x256 PIL

Ported from the NVIDIA Cosmos tokenizer (Apache-2.0, Copyright (c) 2025 NVIDIA
CORPORATION & AFFILIATES), with decoder channels = 192. Its 2D layers adapt the
CompVis stable-diffusion autoencoder (MIT, Copyright (c) 2022 Robin Rombach,
Patrick Esser and contributors), and its FSQ quantizer adapts
lucidrains/vector-quantize-pytorch (MIT, Copyright (c) 2020 Phil Wang).

Preprocessing is the GPIC-256 oracle protocol used to pretokenize the training
and eval corpora: center-crop the square along the longer edge, one bicubic
resize to 256, scale to [-1, 1]. Encoding in bfloat16 reproduces the corpus
pretokenizer; decoding defaults to float32.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

HF_REPO = "radicalnumerics/Halcyon-Tokenizers"
HF_WEIGHTS = "image_tokenizer/di8x8_242172_ema.pt"

IMAGE_SIZE = 256
GRID = 32

# discrete_image_8x8_360p from the training repo.
CONFIG = dict(
    attn_resolutions=[],
    channels=128,
    decoder_channels=192,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    spatial_compression=8,
    num_res_blocks=2,
    out_channels=3,
    resolution=256,
    patch_size=4,
    patch_method="haar",
    z_channels=256,
    num_groups=32,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
)

_HAAR = torch.tensor([0.7071067811865476, 0.7071067811865476])


def _normalize(channels: int, num_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True
    )


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# --- Haar patching -------------------------------------------------------------


class Patcher(nn.Module):
    """log2(patch_size) Haar DWTs: (B, C, H, W) -> (B, C*p^2, H/p, W/p)."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.register_buffer("wavelets", _HAAR.clone(), persistent=True)
        self.register_buffer("_arange", torch.arange(2), persistent=True)
        self.levels = int(math.log2(patch_size))

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        dtype, h, g = x.dtype, self.wavelets, x.shape[1]
        n = h.shape[0]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode="reflect").to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))
        return torch.cat([xll, xlh, xhl, xhh], dim=1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.levels):
            x = self._dwt(x)
        return x


class UnPatcher(nn.Module):
    """Inverse of :class:`Patcher`: (B, C*p^2, H, W) -> (B, C, H*p, W*p)."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.register_buffer("wavelets", _HAAR.clone(), persistent=True)
        self.register_buffer("_arange", torch.arange(2), persistent=True)
        self.levels = int(math.log2(patch_size))

    def _idwt(self, x: torch.Tensor) -> torch.Tensor:
        dtype, h = x.dtype, self.wavelets
        n, g = h.shape[0], x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)
        pv, ph = (n - 2, 0), (0, n - 2)
        yl = F.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=pv
        )
        yl += F.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=pv
        )
        yh = F.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=pv
        )
        yh += F.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=pv
        )
        y = F.conv_transpose2d(yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=ph)
        y += F.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=ph
        )
        return y * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.levels):
            x = self._idwt(x)
        return x


# --- Conv blocks ---------------------------------------------------------------


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None, num_groups: int):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _normalize(in_channels, num_groups)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = _normalize(out_channels, num_groups)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_silu(self.norm1(x)))
        h = self.conv2(self.dropout(_silu(self.norm2(h))))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.norm = _normalize(channels, num_groups)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q, k, v = (t.reshape(b, 1, c, h * w).transpose(2, 3) for t in (q, k, v))
        h_ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        h_ = h_.transpose(2, 3).reshape(b, c, h, w)
        return x + self.proj_out(h_)


def _blocks(in_ch, out_ch, n_blocks, use_attn, num_groups):
    blocks = nn.ModuleList(
        ResnetBlock(in_ch if i == 0 else out_ch, out_ch, num_groups)
        for i in range(n_blocks)
    )
    attns = nn.ModuleList(
        AttnBlock(out_ch, num_groups) if use_attn else nn.Identity()
        for _ in range(n_blocks)
    )
    return blocks, attns


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, use_attn, down, num_groups):
        super().__init__()
        self.blocks, self.attns = _blocks(in_ch, out_ch, n_blocks, use_attn, num_groups)
        self.downsample = Downsample(out_ch) if down else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, attn in zip(self.blocks, self.attns, strict=True):
            x = attn(block(x))
        return self.downsample(x)


class DecoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, use_attn, up, num_groups):
        super().__init__()
        self.blocks, self.attns = _blocks(in_ch, out_ch, n_blocks, use_attn, num_groups)
        self.upsample = Upsample(out_ch) if up else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, attn in zip(self.blocks, self.attns, strict=True):
            x = attn(block(x))
        return self.upsample(x)


class Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch, mult, ng = cfg["channels"], cfg["channels_mult"], cfg["num_groups"]
        p = cfg["patch_size"]
        n_down = int(math.log2(cfg["spatial_compression"] // p))
        self.patcher = Patcher(p)
        self.conv_in = nn.Conv2d(
            cfg["in_channels"] * p**2, ch, kernel_size=3, padding=1
        )
        self.stages = nn.ModuleList()
        res, cur = cfg["resolution"] // p, ch
        for i, m in enumerate(mult):
            self.stages.append(
                EncoderStage(
                    cur,
                    ch * m,
                    cfg["num_res_blocks"],
                    res in cfg["attn_resolutions"],
                    i < n_down,
                    ng,
                )
            )
            cur = ch * m
            if i < n_down:
                res //= 2
        self.mid_block_1 = ResnetBlock(cur, None, ng)
        self.mid_attn = AttnBlock(cur, ng)
        self.mid_block_2 = ResnetBlock(cur, None, ng)
        self.norm_out = _normalize(cur, ng)
        self.conv_out = nn.Conv2d(cur, cfg["z_channels"], kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(self.patcher(x))
        for stage in self.stages:
            x = stage(x)
        x = self.mid_block_2(self.mid_attn(self.mid_block_1(x)))
        return self.conv_out(_silu(self.norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ch, mult, ng = cfg["decoder_channels"], cfg["channels_mult"], cfg["num_groups"]
        p = cfg["patch_size"]
        n_up = int(math.log2(cfg["spatial_compression"] // p))
        res = cfg["resolution"] // cfg["spatial_compression"]
        cur = ch * mult[-1]
        self.conv_in = nn.Conv2d(cfg["z_channels"], cur, kernel_size=3, padding=1)
        self.mid_block_1 = ResnetBlock(cur, None, ng)
        self.mid_attn = AttnBlock(cur, ng)
        self.mid_block_2 = ResnetBlock(cur, None, ng)
        self.stages = nn.ModuleList()
        for j, m in enumerate(reversed(mult)):
            self.stages.append(
                DecoderStage(
                    cur,
                    ch * m,
                    cfg["num_res_blocks"] + 1,
                    res in cfg["attn_resolutions"],
                    j < n_up,
                    ng,
                )
            )
            cur = ch * m
            if j < n_up:
                res *= 2
        self.norm_out = _normalize(cur, ng)
        self.conv_out = nn.Conv2d(
            cur, cfg["out_channels"] * p**2, kernel_size=3, padding=1
        )
        self.unpatcher = UnPatcher(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mid_block_2(self.mid_attn(self.mid_block_1(self.conv_in(z))))
        for stage in self.stages:
            x = stage(x)
        return self.unpatcher(self.conv_out(_silu(self.norm_out(x))))


# --- FSQ ----------------------------------------------------------------------


class FSQuantizer(nn.Module):
    """Finite scalar quantization (arXiv 2309.15505) with an implicit codebook."""

    def __init__(self, levels: list[int]):
        super().__init__()
        self.dtype = torch.bfloat16  # output dtype; set by ImageTokenizer
        self.register_buffer(
            "_levels", torch.tensor(levels, dtype=torch.int32), persistent=True
        )
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", basis, persistent=True)
        self.codebook_dim = len(levels)
        self.codebook_size = int(self._levels.prod().item())
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        codebook = self.indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", codebook, persistent=True)

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, H, W) indices -> (B, 6, H, W) codes in [-1, 1]; 1-D -> (N, 6)."""
        is_image = indices.ndim >= 3
        codes = (indices.unsqueeze(-1) // self._basis) % self._levels
        half = self._levels // 2
        codes = (codes - half) / half
        if is_image:
            codes = codes.movedim(-1, 1)
        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 6, H, W) latents -> (indices (B, H, W) int32, codes (B, 6, H, W))."""
        b, d, h, w = z.shape
        z = z.movedim(1, -1).reshape(b, h * w, 1, d)
        half = self._levels // 2
        # Straight-through form kept verbatim: z + (round(z) - z) is not always
        # bit-equal to round(z), and the index cast below truncates.
        bz = self._bound(z)
        codes = (bz + (bz.round() - bz).detach()) / half
        indices = (
            ((codes * half + half).float() * self._basis).sum(dim=-1).to(torch.int32)
        )
        out = codes.reshape(b, h, w, d).movedim(-1, 1)
        return indices.reshape(b, h, w), out.to(self.dtype)


# --- Tokenizer ----------------------------------------------------------------


class ImageTokenizer(nn.Module):
    """DI8x8: 256x256 RGB in [-1, 1] <-> 32x32 grid of indices into 64,000 FSQ codes."""

    def __init__(self, cfg: dict = CONFIG):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quant_conv = nn.Conv2d(cfg["z_channels"], cfg["embedding_dim"], 1)
        self.post_quant_conv = nn.Conv2d(cfg["embedding_dim"], cfg["z_channels"], 1)
        self.quantizer = FSQuantizer(cfg["levels"])

    @classmethod
    def from_pretrained(
        cls,
        weights: str | Path | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> ImageTokenizer:
        """Load weights from a local ``.pt`` or, by default, the Halcyon HF repo."""
        if weights is None or not Path(weights).exists():
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(HF_REPO, str(weights) if weights else HF_WEIGHTS)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        tok = cls()
        tok.load_state_dict(state, strict=True)
        tok.quantizer.dtype = dtype
        return tok.eval().requires_grad_(False).to(device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self.quant_conv.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.quant_conv.weight.dtype

    @staticmethod
    def preprocess(images: list[Image.Image], size: int = IMAGE_SIZE) -> torch.Tensor:
        """PIL images -> (B, 3, size, size) in [-1, 1], GPIC oracle center crop."""
        out = []
        for pil in images:
            pil = pil.convert("RGB")
            w, h = pil.size
            c = min(w, h)
            pil = pil.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
            pil = pil.resize((size, size), Image.Resampling.BICUBIC)
            arr = (np.asarray(pil, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
            out.append(torch.from_numpy(arr))
        return torch.stack(out)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 256, 256) in [-1, 1] -> (B, 32, 32) int64 codebook indices."""
        x = x.to(self.device, self.dtype)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.dtype == torch.bfloat16
        ):
            indices, _ = self.quantizer(self.quant_conv(self.encoder(x)))
        return indices.long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, 32, 32) or (B, 1024) codebook indices -> (B, 3, 256, 256) in [-1, 1]."""
        codes = torch.as_tensor(codes).to(self.device, torch.int64)
        if codes.ndim == 2:
            side = math.isqrt(codes.shape[1])
            codes = codes.reshape(codes.shape[0], side, side)
        hi, lo = int(codes.max()), int(codes.min())
        if lo < 0 or hi >= self.quantizer.codebook_size:
            raise ValueError(
                f"codes span [{lo}, {hi}], outside the "
                f"{self.quantizer.codebook_size}-entry codebook; pass codebook "
                "indices, not LM token ids"
            )
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.dtype == torch.bfloat16
        ):
            quant = self.quantizer.indices_to_codes(codes).to(self.dtype)
            return self.decoder(self.post_quant_conv(quant)).float()

    @staticmethod
    def to_pil(x: torch.Tensor) -> list[Image.Image]:
        """(B, 3, H, W) in [-1, 1] -> list of RGB PIL images."""
        arr = ((x.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        return [Image.fromarray(a) for a in arr.permute(0, 2, 3, 1).cpu().numpy()]
