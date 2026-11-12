from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from xurcroh.fortress.arithmetic import (
    PRIME_MODULUS,
    RING_MODULUS,
    QuantizedTensor,
    freivalds_challenge,
    seeded_split_secret,
    split_secret,
    symmetric_quantize,
)
from xurcroh.fortress.grid import WeightShare
from xurcroh.fortress.storage import DispersedShard, disperse_shard


@dataclass(frozen=True)
class StoredWeightShare:
    shape: tuple[int, ...]
    bundle: DispersedShard


@dataclass(frozen=True)
class CompiledOperation:
    quantized: QuantizedTensor
    shares: dict[int, WeightShare]
    storage: dict[int, StoredWeightShare]


def compile_operation(
    weight: torch.Tensor,
    *,
    threshold: int,
    slice_count: int,
    inner_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: bytes | None = None,
) -> CompiledOperation:
    if inner_transform is not None:
        weight = inner_transform(weight)
    quantized = symmetric_quantize(weight, bits=4)
    if seed is None:
        grid = split_secret(quantized.values, count=2, modulus=RING_MODULUS)
    else:
        grid = seeded_split_secret(quantized.values, seed=seed, count=2)
    shares: dict[int, WeightShare] = {}
    storage: dict[int, StoredWeightShare] = {}
    for index, matrix in enumerate(grid.shares):
        challenge_values = [secrets.randbelow(PRIME_MODULUS) for _ in range(matrix.shape[0])]
        challenge = freivalds_challenge(matrix, r=torch.tensor(challenge_values, dtype=torch.int64))
        shares[index] = WeightShare(
            weight_index=index,
            matrix=matrix,
            challenge=challenge,
            provider_id=f"weight-share-{index}",
        )
        payload = matrix.detach().cpu().numpy().tobytes()
        storage[index] = StoredWeightShare(
            shape=tuple(matrix.shape),
            bundle=disperse_shard(payload, threshold=threshold, slice_count=slice_count),
        )
    return CompiledOperation(quantized=quantized, shares=shares, storage=storage)


def reap_prune_experts(
    experts: Mapping[str, torch.Tensor],
    saliency_scores: Mapping[str, float],
    *,
    keep_fraction: float,
) -> dict[str, torch.Tensor]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be between zero and one")
    if set(experts) != set(saliency_scores):
        raise ValueError("every expert requires a saliency score")
    keep = max(1, (len(experts) * keep_fraction).__ceil__())
    ranked = sorted(experts, key=lambda name: saliency_scores[name], reverse=True)
    return {name: experts[name] for name in ranked[:keep]}
