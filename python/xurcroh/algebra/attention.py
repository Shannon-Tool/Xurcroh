from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from xurcroh.algebra.orthogonal import sample_orthogonal

__all__ = [
    "AttentionWeights",
    "conjugate_attention",
    "reference_attention",
    "reference_attention_cached",
    "rope_commuting_pair",
    "sample_orthogonal",
    "validate_rope_commuting",
]


@dataclass(frozen=True)
class AttentionWeights:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None


def _rope_head(
    head_dimension: int,
    angles: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for angle, scale in zip(angles, scales, strict=True):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack([torch.stack([cosine, -sine]), torch.stack([sine, cosine])])
        blocks.append(scale * rotation)
    return cast(torch.Tensor, torch.block_diag(*blocks))  # type: ignore[no-untyped-call]


def rope_commuting_pair(
    query_heads: int,
    kv_heads: int,
    head_dimension: int,
    *,
    seed: int,
    minimum_scale: float = 0.6,
    maximum_scale: float = 1.6,
    interleaved: bool = True,
    quarter_turns: bool = False,
    shared_heads: bool = False,
    preserve_rms_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_heads % kv_heads or head_dimension % 2:
        raise ValueError("GQA heads must divide evenly and RoPE head dimension must be even")
    if not 0 < minimum_scale <= maximum_scale:
        raise ValueError("contragredient scales must be positive and ordered")
    if preserve_rms_norm and (minimum_scale != 1 or maximum_scale != 1):
        raise ValueError("RMS-safe RoPE transforms cannot use contragredient scaling")
    generator = torch.Generator().manual_seed(seed)
    q_blocks: list[torch.Tensor] = []
    k_blocks: list[torch.Tensor] = []
    for _ in range(kv_heads):
        if shared_heads and q_blocks and k_blocks:
            k_blocks.append(k_blocks[0])
            q_blocks.extend(q_blocks[0] for _ in range(query_heads // kv_heads))
            continue
        if preserve_rms_norm:
            pairs = torch.randperm(head_dimension // 2, generator=generator)
            order = torch.repeat_interleave(pairs, 2) * 2 + torch.tensor([0, 1]).repeat(
                head_dimension // 2
            )
            q_head = torch.eye(head_dimension, dtype=torch.float64)[order][:, order]
            k_head = q_head
        else:
            if quarter_turns:
                angles = torch.randint(
                    0, 4, (head_dimension // 2,), generator=generator, dtype=torch.int64
                ).to(torch.float64) * (torch.pi / 2)
            else:
                angles = torch.rand(head_dimension // 2, generator=generator, dtype=torch.float64)
                angles = angles * (2 * torch.pi)
            scales = torch.empty(head_dimension // 2, dtype=torch.float64)
            scales.uniform_(minimum_scale, maximum_scale, generator=generator)
            q_head = _rope_head(head_dimension, angles, scales)
            k_head = _rope_head(head_dimension, angles, scales.reciprocal())
        k_blocks.append(k_head)
        q_blocks.extend(q_head for _ in range(query_heads // kv_heads))
    if not interleaved:
        half = head_dimension // 2
        input_order = torch.stack(
            (torch.arange(half), torch.arange(half, head_dimension)), dim=1
        ).reshape(-1)
        output_order = torch.argsort(input_order)
        q_blocks = [block[output_order][:, output_order] for block in q_blocks]
        k_blocks = [block[output_order][:, output_order] for block in k_blocks]
    q_transform = cast(torch.Tensor, torch.block_diag(*q_blocks))  # type: ignore[no-untyped-call]
    k_transform = cast(torch.Tensor, torch.block_diag(*k_blocks))  # type: ignore[no-untyped-call]
    return q_transform, k_transform


def validate_rope_commuting(matrix: torch.Tensor, tolerance: float = 1e-7) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
        raise ValueError("RoPE transform must be a square matrix of 2x2 rotation blocks")
    blocks = matrix.reshape(-1, 2, 2)
    off_diagonal = matrix.clone()
    for index in range(blocks.shape[0]):
        off_diagonal[index * 2 : index * 2 + 2, index * 2 : index * 2 + 2] = 0
    if (
        torch.count_nonzero(off_diagonal)
        or not torch.allclose(blocks[:, 0, 0], blocks[:, 1, 1], atol=tolerance, rtol=0)
        or not torch.allclose(blocks[:, 0, 1], -blocks[:, 1, 0], atol=tolerance, rtol=0)
    ):
        raise ValueError("RoPE transform must contain scaled 2x2 rotation blocks")


def apply_rope(
    values: torch.Tensor,
    base: float = 10_000,
    *,
    interleaved: bool = True,
    position_offset: int = 0,
) -> torch.Tensor:
    tokens, _, head_dimension = values.shape
    frequencies = base ** (
        -torch.arange(0, head_dimension, 2, dtype=torch.float64) / head_dimension
    )
    positions = torch.arange(position_offset, position_offset + tokens, dtype=torch.float64)
    angles = positions[:, None] * frequencies[None, :]
    angles = angles.to(device=values.device, dtype=values.dtype)
    if interleaved:
        pairs = values.reshape(tokens, values.shape[1], head_dimension // 2, 2)
        x, y = pairs[..., 0], pairs[..., 1]
    else:
        x = values[..., : head_dimension // 2]
        y = values[..., head_dimension // 2 :]
    rotated = torch.stack(
        [
            x * torch.cos(angles)[:, None] - y * torch.sin(angles)[:, None],
            x * torch.sin(angles)[:, None] + y * torch.cos(angles)[:, None],
        ],
        dim=-1,
    )
    if interleaved:
        return rotated.reshape_as(values)
    return torch.cat(rotated.unbind(dim=-1), dim=-1)


def reference_attention_cached(
    hidden: torch.Tensor,
    weights: AttentionWeights,
    query_heads: int,
    kv_heads: int,
    *,
    past: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_offset: int | None = None,
    rotary: bool = True,
    interleaved: bool = True,
    norm_epsilon: float = 1e-6,
    rope_base: float = 10_000,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    tokens = hidden.shape[0]
    head_dimension = weights.q.shape[0] // query_heads
    q = (hidden @ weights.q.T).reshape(tokens, query_heads, head_dimension)
    k = (hidden @ weights.k.T).reshape(tokens, kv_heads, head_dimension)
    v = (hidden @ weights.v.T).reshape(tokens, kv_heads, head_dimension)
    if weights.q_norm is not None:
        q = torch.nn.functional.rms_norm(q, (head_dimension,), weights.q_norm, norm_epsilon)
    if weights.k_norm is not None:
        k = torch.nn.functional.rms_norm(k, (head_dimension,), weights.k_norm, norm_epsilon)
    if position_offset is None:
        position_offset = 0 if past is None else past[0].shape[0]
    if rotary:
        q = apply_rope(q, base=rope_base, interleaved=interleaved, position_offset=position_offset)
        k = apply_rope(k, base=rope_base, interleaved=interleaved, position_offset=position_offset)
    if past is not None:
        k = torch.cat([past[0], k], dim=0)
        v = torch.cat([past[1], v], dim=0)
    cache = (k, v)
    groups = query_heads // kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / head_dimension**0.5
    total = k.shape[0]
    causal = torch.triu(
        torch.ones(tokens, total, dtype=torch.bool, device=hidden.device),
        total - tokens + 1,
    )
    scores = scores.masked_fill(causal, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.einsum("hts,shd->thd", probabilities, v)
    output = context.reshape(tokens, query_heads * head_dimension) @ weights.o.T
    return output, cache


def reference_attention(
    hidden: torch.Tensor,
    weights: AttentionWeights,
    query_heads: int,
    kv_heads: int,
    *,
    rotary: bool,
    interleaved: bool = True,
    norm_epsilon: float = 1e-6,
    rope_base: float = 10_000,
) -> torch.Tensor:
    return reference_attention_cached(
        hidden,
        weights,
        query_heads,
        kv_heads,
        rotary=rotary,
        interleaved=interleaved,
        norm_epsilon=norm_epsilon,
        rope_base=rope_base,
    )[0]


def conjugate_attention(
    weights: AttentionWeights,
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    q_transform: torch.Tensor,
    k_transform: torch.Tensor,
    value_transform: torch.Tensor,
    output_value_inverse: torch.Tensor,
) -> AttentionWeights:
    return AttentionWeights(
        q=q_transform @ weights.q @ input_basis.T,
        k=k_transform @ weights.k @ input_basis.T,
        v=value_transform @ weights.v @ input_basis.T,
        o=output_basis @ weights.o @ output_value_inverse,
        q_norm=None
        if weights.q_norm is None
        else q_transform[: weights.q_norm.shape[0], : weights.q_norm.shape[0]] @ weights.q_norm,
        k_norm=None
        if weights.k_norm is None
        else k_transform[: weights.k_norm.shape[0], : weights.k_norm.shape[0]] @ weights.k_norm,
    )
