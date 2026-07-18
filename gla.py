# ============================================================================
# Original algorithm from:
#   "Gated Linear Attention Transformers with Hardware-Efficient Training"
#   (Yang et al., 2023) - https://arxiv.org/abs/2312.06635
#
# ============================================================================

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rotary Embedding
def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Stack [-x2, x1] interleaved: [-x2_0, x1_0, -x2_1, x1_1, ...]
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

    # Expand cos/sin to match x: [seq_len, 1, ro_dim]
    if not interleaved:
        # [seq_len, d] -> [seq_len, 1, 2d] by repeating each element
        cos = cos.unsqueeze(-2)  # [seq_len, 1, d]
        sin = sin.unsqueeze(-2)
        cos = cos.repeat(1, 1, 2)  # [seq_len, 1, 2d]
        sin = sin.repeat(1, 1, 2)
    else:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        # For interleaved: [d] -> [d*2] by repeating each element twice
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

        # Slice the cached cos/sin for the relevant positions
        cos = self._cos_cached[seqlen_offset:total_seqlen]  # [seqlen, dim/2]
        sin = self._sin_cached[seqlen_offset:total_seqlen]  # [seqlen, dim/2]

        q_rotated = apply_rotary_embedding(q, cos, sin, self.interleaved)
        k_rotated = apply_rotary_embedding(k, cos, sin, self.interleaved)

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, base={self.base}, "
            f"interleaved={self.interleaved}, "
            f"pos_idx_in_fp32={self.pos_idx_in_fp32}"
        )


# GLA Operations — Pure PyTorch
def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Recurrent (step-by-step) Gated Linear Attention in pure PyTorch.

    For each timestep t:
        S_t = diag(exp(gk_t)) * S_{t-1} + k_t^T v_t
        o_t = q_t @ S_t

    Args:
        q: Queries [B, T, H, K]
        k: Keys [B, T, H, K]
        v: Values [B, T, H, V]
        gk: Log-space forget gates [B, T, H, K]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state

    Returns:
        (output [B, T, H, V], final_state [B, H, K, V] or None)
    """
    orig_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Work in float32 for numerical stability
    q = q.float() * scale
    k = k.float()
    v = v.float()
    gk = gk.float()

    # Transpose to [B, H, T, dim] for easier per-head processing
    q = q.transpose(1, 2)   # [B, H, T, K]
    k = k.transpose(1, 2)   # [B, H, T, K]
    v = v.transpose(1, 2)   # [B, H, T, V]
    gk = gk.transpose(1, 2) # [B, H, T, K]

    # Initialize state: [B, H, K, V]
    if initial_state is not None:
        S = initial_state.float()
    else:
        S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    outputs = []

    for t in range(T):
        # Gate decay for this timestep: [B, H, K]
        gate = gk[:, :, t, :].exp()  # exp(log_gate) = gate value in [0,1]

        # Apply gate to state: element-wise along K dim
        # S: [B, H, K, V], gate: [B, H, K] -> [B, H, K, 1]
        S = S * gate.unsqueeze(-1)

        # Outer product update: k_t^T @ v_t -> [B, H, K, V]
        # k_t: [B, H, K], v_t: [B, H, V]
        k_t = k[:, :, t, :]  # [B, H, K]
        v_t = v[:, :, t, :]  # [B, H, V]
        S = S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        # Output: q_t @ S -> [B, H, V]
        q_t = q[:, :, t, :]  # [B, H, K]
        o_t = torch.einsum('bhk,bhkv->bhv', q_t, S)
        outputs.append(o_t)

    # Stack along time dim: [B, H, T, V]
    output = torch.stack(outputs, dim=2)

    # Transpose back: [B, T, H, V]
    output = output.transpose(1, 2).to(orig_dtype)

    final_state = None
    if output_final_state:
        final_state = S.to(orig_dtype)

    return output, final_state


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunked parallel Gated Linear Attention in pure PyTorch.

    Algorithm from Yang et al. (2023) Section 3.2:
    1. Split sequence into chunks of size C
    2. Intra-chunk: compute gated attention within each chunk
    3. Inter-chunk: propagate hidden state across chunks
    4. Combine intra + inter results

    Args:
        q: Queries [B, T, H, K]
        k: Keys [B, T, H, K]
        v: Values [B, T, H, V]
        gk: Log-space forget gates [B, T, H, K]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state
        chunk_size: Size of each chunk (default: 64)

    Returns:
        (output [B, T, H, V], final_state [B, H, K, V] or None)
    """
    orig_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if scale is None:
        scale = K ** -0.5

    # Pad sequence length to multiple of chunk_size
    pad_len = (C - T % C) % C
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        gk = F.pad(gk, (0, 0, 0, 0, 0, pad_len))

    T_padded = q.shape[1]
    num_chunks = T_padded // C

    # Work in float32
    q = q.float() * scale
    k = k.float()
    v = v.float()
    gk = gk.float()

    # Reshape into chunks: [B, num_chunks, C, H, dim]
    q = q.reshape(B, num_chunks, C, H, K)
    k = k.reshape(B, num_chunks, C, H, K)
    v = v.reshape(B, num_chunks, C, H, V)
    gk = gk.reshape(B, num_chunks, C, H, K)

    # Transpose for per-head processing: [B, H, num_chunks, C, dim]
    q = q.permute(0, 3, 1, 2, 4)   # [B, H, NC, C, K]
    k = k.permute(0, 3, 1, 2, 4)   # [B, H, NC, C, K]
    v = v.permute(0, 3, 1, 2, 4)   # [B, H, NC, C, V]
    gk = gk.permute(0, 3, 1, 2, 4) # [B, H, NC, C, K]


    # Cumulative gate within each chunk: G_cum[c, i] = sum_{j=0}^{i} gk[c,j]
    
    gk_cumsum = gk.cumsum(dim=3)  # [B, H, NC, C, K]

    # Intra-chunk attention
    # A[i,j] = sum_k q[i,k] * k[j,k] * exp(G_cum[i,k] - G_cum[j,k])
    # This is a [C, C] attention matrix per chunk per head

    # For numerical stability, compute in log-space relative to each query
    # q_scaled[i] = q[i] * exp(G_cum[i])
    # k_scaled[j] = k[j] * exp(-G_cum[j])
    # A[i,j] = q_scaled[i] @ k_scaled[j]^T

    # But we need to handle the exp carefully to avoid overflow.
    # Instead, compute: A[i,j] = (q[i] * k[j]) * exp(G_cum[i] - G_cum[j])
    # where the sum over K is done element-wise

    # Attention scores: [B, H, NC, C, C]
    # A_ij = sum_k q_ik * k_jk * exp(gcum_ik - gcum_jk)
    # = sum_k (q_ik * exp(gcum_ik)) * (k_jk * exp(-gcum_jk))

    q_decay = q * gk_cumsum.exp()       # [B, H, NC, C, K]
    k_decay = k * (-gk_cumsum).exp()     # [B, H, NC, C, K]

    # A = q_decay @ k_decay^T -> [B, H, NC, C, C]
    attn = torch.matmul(q_decay, k_decay.transpose(-1, -2))

    # Apply causal mask (lower triangular within each chunk)
    causal_mask = torch.tril(
        torch.ones(C, C, device=q.device, dtype=torch.bool)
    )
    attn = attn.masked_fill(~causal_mask, 0.0)

    # Intra-chunk output: A @ V -> [B, H, NC, C, V]
    o_intra = torch.matmul(attn, v)

    # Inter-chunk state propagation
    # For each chunk c, we need the hidden state S_c that summarizes
    # all previous chunks.
    #
    # State update:
    #   S_{c+1} = diag(exp(G_total_c)) * S_c + sum_i k_i^T v_i * exp(G_total_c - G_cum_i)
    # where G_total_c = G_cum[c, C-1] (total gate sum for chunk c)

    # Total gate decay per chunk: [B, H, NC, K]
    gk_total = gk_cumsum[:, :, :, -1, :]  # last position's cumsum

    # Key-value contribution per position:
    # kv[i] = k[i]^T v[i] * exp(G_total - G_cum[i])
    # We accumulate this per chunk

    # For the state update, we need:
    # k_state[i] = k[i] * exp(G_total_c - G_cum[i,k])
    # Then chunk_kv = sum_i k_state[i]^T @ v[i]

    # exp(G_total - G_cum): [B, H, NC, C, K]
    decay_within = (gk_total.unsqueeze(3) - gk_cumsum).exp()

    # k_weighted: k * decay_within -> [B, H, NC, C, K]
    k_weighted = k * decay_within

    # chunk_kv: sum over positions within chunk of k_weighted^T @ v
    # k_weighted: [B, H, NC, C, K], v: [B, H, NC, C, V]
    # k_weighted^T @ v = sum_i k_weighted[i]^T v[i] = [B, H, NC, K, V]
    chunk_kv = torch.matmul(
        k_weighted.transpose(-1, -2),  # [B, H, NC, K, C]
        v                              # [B, H, NC, C, V]
    )  # [B, H, NC, K, V]

    # Now propagate states across chunks
    # S_0 = initial_state (or zeros)
    # S_{c+1} = diag(exp(G_total_c)) * S_c + chunk_kv_c

    if initial_state is not None:
        state = initial_state.float()  # [B, H, K, V]
    else:
        state = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    # Collect states for each chunk (S before processing that chunk)
    states = []

    for c in range(num_chunks):
        states.append(state.clone())

        # Decay the state by the total gate of this chunk
        # gk_total[:, :, c, :] -> [B, H, K]
        total_decay = gk_total[:, :, c, :].exp()  # [B, H, K]

        # Update state
        state = state * total_decay.unsqueeze(-1) + chunk_kv[:, :, c, :, :]

    # Stack states: [B, H, NC, K, V]
    states = torch.stack(states, dim=2)

    # Inter-chunk contribution to output:
    # o_inter[c, i] = q[c,i] * exp(G_cum[c,i]) @ S_c
    # q_decay already = q * exp(G_cum): [B, H, NC, C, K]
    # S_c: [B, H, NC, K, V]
    o_inter = torch.matmul(q_decay, states)  # [B, H, NC, C, V]

    # Combine
    output = o_intra + o_inter  # [B, H, NC, C, V]

    # Reshape: [B, H, NC, C, V] -> [B, H, T_padded, V]
    output = output.reshape(B, H, T_padded, V)

    # Trim padding back to original sequence length
    orig_T = T_padded - pad_len
    if pad_len > 0:
        output = output[:, :, :orig_T, :]

    # Transpose back to [B, T, H, V]
    output = output.transpose(1, 2).to(orig_dtype)

    final_state_out = None
    if output_final_state:
        final_state_out = state.to(orig_dtype)

    return output, final_state_out


# Alias: fused_chunk_gla has the same semantics, just different kernel strategy
# In pure PyTorch they are equivalent
fused_chunk_gla = chunk_gla



# GatedLinearAttention Module
class RMSNormGated(nn.Module):
    """
    RMSNorm with gated activation.
    Replaces fla.modules.FusedRMSNormGated with a pure PyTorch equivalent.

    Computes: output = RMSNorm(x) * activation(z)
    where x and z are the two halves of the input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "swish",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

        if activation == "swish" or activation == "silu":
            self.act_fn = F.silu
        elif activation == "sigmoid":
            self.act_fn = torch.sigmoid
        else:
            self.act_fn = F.silu  # default to swish

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor to normalize [..., hidden_size]
            z: Gate tensor [..., hidden_size]

        Returns:
            Normalized and gated output [..., hidden_size]
        """
        # RMSNorm
        input_dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_normed = x_normed.to(input_dtype)

        # Apply learnable weight and gate
        return x_normed * self.weight * self.act_fn(z)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


class GatedLinearAttention(nn.Module):
    """
    Pure PyTorch implementation of Gated Linear Attention.

    Drop-in replacement for fla.layers.GatedLinearAttention.

    From "Gated Linear Attention Transformers with Hardware-Efficient Training"
    (Yang et al., 2023) - https://arxiv.org/abs/2312.06635

    Args:
        mode: GLA kernel mode. One of 'chunk', 'fused_recurrent', 'fused_chunk'.
        hidden_size: Model hidden dimension.
        expand_k: Expansion ratio for key dimension.
        expand_v: Expansion ratio for value dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (for GQA). Defaults to num_heads.
        feature_map: Feature map function (not used in default config).
        use_short_conv: Whether to use short convolutions (not implemented).
        conv_size: Kernel size for short conv (not implemented).
        conv_bias: Bias for short conv (not implemented).
        use_output_gate: Whether to use output gating.
        gate_fn: Activation function for the output gate.
        elementwise_affine: Whether LayerNorm uses learnable params.
        norm_eps: Epsilon for normalization layers.
        gate_logit_normalizer: Normalizer for gate logits.
        gate_low_rank_dim: Low-rank dimension for gate projection.
        clamp_min: Minimum clamp value for gate logits.
        fuse_norm: Whether to fuse norm (uses RMSNormGated).
        layer_idx: Layer index (for caching).
    """

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

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Output gate projection
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Gate key projection (low-rank)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

        # Output projection
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Normalization: FusedRMSNormGated or standard RMSNorm
        if self.use_output_gate and self.fuse_norm:
            self.g_norm_swish_gate = RMSNormGated(
                self.head_v_dim,
                eps=norm_eps,
                activation=self.gate_fn,
            )
        else:
            self.g_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)

        # Select the GLA operation based on mode
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
        """
        Forward pass for Gated Linear Attention.

        Args:
            hidden_states: Input tensor [B, T, hidden_size]
            attention_mask: Not used (kept for API compat)
            past_key_values: Cache object (kept for API compat)
            use_cache: Whether to return cache
            output_attentions: Not used

        Returns:
            Tuple of (output [B, T, hidden_size], attention_weights, past_key_values)
        """
        B, T, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Compute gate
        gk = self.gk_proj(hidden_states)

        # Output gate
        if self.use_output_gate:
            g = self.g_proj(hidden_states)

        # Reshape to multi-head: [B, T, H, D]
        q = q.view(B, T, self.num_heads, self.head_k_dim)
        k = k.view(B, T, self.num_kv_heads, self.key_dim_per_group)
        v = v.view(B, T, self.num_kv_heads, self.value_dim_per_group)
        gk = gk.view(B, T, self.num_kv_heads, self.key_dim_per_group)

        # Apply gate normalization: logsigmoid / normalizer
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = gk.clamp(min=self.clamp_min)

        # Handle GQA (grouped query attention) by repeating K, V, gk
        if self.num_kv_groups > 1:
            # Repeat k, v, gk for each group
            k = k.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.key_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_k_dim)
            v = v.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.value_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_v_dim)
            gk = gk.unsqueeze(3).expand(
                B, T, self.num_kv_heads, self.num_kv_groups, self.key_dim_per_group
            ).reshape(B, T, self.num_heads, self.head_k_dim)

        # Determine initial state from cache if available
        initial_state = None
        if past_key_values is not None and hasattr(past_key_values, 'get'):
            try:
                initial_state = past_key_values.get(self.layer_idx)
            except Exception:
                pass

        # Run the GLA operation
        if self.mode == "fused_recurrent":
            o, final_state = self.chunk_fn(
                q=q, k=k, v=v, gk=gk,
                initial_state=initial_state,
                output_final_state=use_cache,
            )
        else:
            o, final_state = self.chunk_fn(
                q=q, k=k, v=v, gk=gk,
                initial_state=initial_state,
                output_final_state=use_cache,
            )

        # Apply output gate with fused norm
        if self.use_output_gate:
            g = g.view(B, T, self.num_heads, self.head_v_dim)

            if self.fuse_norm:
                o = self.g_norm_swish_gate(o, g)
            else:
                o = self.g_norm(o) * F.silu(g)
        else:
            o = self.g_norm(o)

        # Reshape and project output
        o = o.reshape(B, T, self.value_dim)
        o = self.o_proj(o)

        return o, None, past_key_values

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"key_dim={self.key_dim}, value_dim={self.value_dim}"
        )        