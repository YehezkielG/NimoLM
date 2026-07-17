# Original algorithm from:
#   "Gated Linear Attention Transformers with Hardware-Efficient Training"
#   (Yang et al., 2023) - https://arxiv.org/abs/2312.06635

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        stacked = torch.stack((-x2, x1), dim=-1)
        return stacked.reshape(x.shape)


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:

    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], f"Rotary dim {ro_dim} > head dim {x.shape[-1]}"

    if not interleaved:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
    else:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

    x_rot = x[..., :ro_dim]
    x_pass = x[..., ro_dim:]

    x_rotated = x_rot * cos + rotate_half(x_rot, interleaved) * sin

    if x_pass.shape[-1] > 0:
        return torch.cat([x_rotated, x_pass], dim=-1)
    return x_rotated


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        max_seqlen: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.max_seqlen = max_seqlen

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _update_cos_sin_cache(
        self, seqlen: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if seqlen <= self._seq_len_cached and self._cos_cached is not None:
            if self._cos_cached.device == device and self._cos_cached.dtype == dtype:
                return

        self._seq_len_cached = seqlen

        if self.pos_idx_in_fp32:
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(torch.float32)
        else:
            t = torch.arange(seqlen, device=device, dtype=dtype)
            inv_freq = self.inv_freq.to(dtype)

        freqs = torch.outer(t, inv_freq)

        self._cos_cached = freqs.cos().to(dtype)
        self._sin_cached = freqs.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        seqlen = q.shape[1]
        total_seqlen = seqlen + seqlen_offset

        self._update_cos_sin_cache(total_seqlen, q.device, q.dtype)

        assert self._cos_cached is not None and self._sin_cached is not None

        cos = self._cos_cached[seqlen_offset:total_seqlen]
        sin = self._sin_cached[seqlen_offset:total_seqlen]

        q_rotated = apply_rotary_embedding(q, cos, sin, self.interleaved)
        k_rotated = apply_rotary_embedding(k, cos, sin, self.interleaved)

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, base={self.base}, "
            f"interleaved={self.interleaved}, "
            f"pos_idx_in_fp32={self.pos_idx_in_fp32}"
        )
