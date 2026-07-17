# Gated Linear Attention Layer
# "Gated Linear Attention Transformers with Hardware-Efficient Training"
# (Yang et al., 2023) - https://arxiv.org/abs/2312.06635

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nimo_src.modules.norm import RMSNormGated
from nimo_src.modules.gla_ops import chunk_gla, fused_chunk_gla, fused_recurrent_gla


class GatedLinearAttention(nn.Module):

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_heads
        self.value_dim_per_group = self.value_dim // self.num_kv_heads
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_output_gate = use_output_gate
        self.gate_fn = gate_fn
        self.gate_logit_normalizer = gate_logit_normalizer
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx
        self.fuse_norm = fuse_norm

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if self.use_output_gate and self.fuse_norm:
            self.g_norm_swish_gate = RMSNormGated(
                self.head_v_dim,
                eps=norm_eps,
                activation=self.gate_fn,
            )
        else:
            self.g_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)

        if mode == "chunk":
            self.chunk_fn = chunk_gla
        elif mode == "fused_chunk":
            self.chunk_fn = fused_chunk_gla
        elif mode == "fused_recurrent":
            self.chunk_fn = fused_recurrent_gla
        else:
            raise ValueError(f"Unknown GLA mode: {mode}. Choose from: chunk, fused_chunk, fused_recurrent")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:

        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        gk = self.gk_proj(hidden_states)

        if self.use_output_gate:
            g = self.g_proj(hidden_states)

        q = q.view(B, T, self.num_heads, self.head_k_dim)
        k = k.view(B, T, self.num_kv_heads, self.key_dim_per_group)
        v = v.view(B, T, self.num_kv_heads, self.value_dim_per_group)
        gk = gk.view(B, T, self.num_kv_heads, self.key_dim_per_group)

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = gk.clamp(min=self.clamp_min)

        if self.num_kv_groups > 1:
            k = k.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.key_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_k_dim)
            v = v.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.value_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_v_dim)
            gk = gk.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.key_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_k_dim)

        initial_state = None
        if past_key_values is not None and hasattr(past_key_values, 'get'):
            try:
                initial_state = past_key_values.get(self.layer_idx)
            except Exception:
                pass

        o, final_state = self.chunk_fn(
            q=q, k=k, v=v, gk=gk,
            initial_state=initial_state,
            output_final_state=use_cache,
        )

        if self.use_output_gate:
            g = g.view(B, T, self.num_heads, self.head_v_dim)

            if self.fuse_norm:
                o = self.g_norm_swish_gate(o, g)
            else:
                o = self.g_norm(o) * F.silu(g)
        else:
            o = self.g_norm(o)

        o = o.reshape(B, T, self.value_dim)
        o = self.o_proj(o)

        return o, None, past_key_values

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"key_dim={self.key_dim}, value_dim={self.value_dim}"
        )
