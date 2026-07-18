# Gated Linear Attention Operations — Pure PyTorch
# Algorithm from Yang et al. (2023) Section 3.2

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


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
    Recurrent (step-by-step) Gated Linear Attention.

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

    q = q.float() * scale
    k = k.float()
    v = v.float()
    gk = gk.float()

    q = q.transpose(1, 2)   # [B, H, T, K]
    k = k.transpose(1, 2)   # [B, H, T, K]
    v = v.transpose(1, 2)   # [B, H, T, V]
    gk = gk.transpose(1, 2) # [B, H, T, K]

    if initial_state is not None:
        S = initial_state.float()
    else:
        S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    outputs = []

    for t in range(T):
        gate = gk[:, :, t, :].exp()
        S = S * gate.unsqueeze(-1)

        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        S = S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        q_t = q[:, :, t, :]
        o_t = torch.einsum('bhk,bhkv->bhv', q_t, S)
        outputs.append(o_t)

    output = torch.stack(outputs, dim=2)
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
    Chunked parallel Gated Linear Attention.

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

    # Cumulative gate within each chunk
    gk_cumsum = gk.cumsum(dim=3)  # [B, H, NC, C, K]

    # Intra-chunk attention
    # A_ij = sum_k (q_ik * exp(gcum_ik)) * (k_jk * exp(-gcum_jk))
    q_decay = q * torch.exp(torch.clamp(gk_cumsum, max=85.0))       
    k_decay = k * torch.exp(torch.clamp(-gk_cumsum, max=85.0))

    attn = torch.matmul(q_decay, k_decay.transpose(-1, -2))

    # Causal mask (lower triangular within each chunk)
    causal_mask = torch.tril(
        torch.ones(C, C, device=q.device, dtype=torch.bool)
    )
    attn = attn.masked_fill(~causal_mask, 0.0)

    o_intra = torch.matmul(attn, v)

    # Inter-chunk state propagation
    gk_total = gk_cumsum[:, :, :, -1, :]  # [B, H, NC, K]

    decay_diff = gk_total.unsqueeze(3) - gk_cumsum
    decay_within = torch.exp(torch.clamp(decay_diff, max=85.0))
    k_weighted = k * decay_within

    chunk_kv = torch.matmul(
        k_weighted.transpose(-1, -2),  # [B, H, NC, K, C]
        v                              # [B, H, NC, C, V]
    )  # [B, H, NC, K, V]

    if initial_state is not None:
        state = initial_state.float()
    else:
        state = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)

    states = []

    for c in range(num_chunks):
        states.append(state.clone())
        total_decay = gk_total[:, :, c, :].exp()
        state = state * total_decay.unsqueeze(-1) + chunk_kv[:, :, c, :, :]

    states = torch.stack(states, dim=2)

    o_inter = torch.matmul(q_decay, states)  # [B, H, NC, C, V]

    # Combine
    output = o_intra + o_inter  # [B, H, NC, C, V]

    output = output.reshape(B, H, T_padded, V)

    orig_T = T_padded - pad_len
    if pad_len > 0:
        output = output[:, :, :orig_T, :]

    output = output.transpose(1, 2).to(orig_dtype)

    final_state_out = None
    if output_final_state:
        final_state_out = state.to(orig_dtype)

    return output, final_state_out


fused_chunk_gla = chunk_gla
