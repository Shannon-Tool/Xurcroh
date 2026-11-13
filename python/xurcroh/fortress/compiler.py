from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from xurcroh.algebra.modulation import SignedPermutation
from xurcroh.fortress.arithmetic import (
    FIELD_MODULUS,
    PRIME_MODULUS,
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


BasisTransform = torch.Tensor | SignedPermutation


@dataclass(frozen=True)
class CompiledOperation:
    quantized: QuantizedTensor
    shares: dict[int, WeightShare]
    storage: dict[int, StoredWeightShare]
    input_basis: BasisTransform | None = field(default=None, repr=False)
    output_basis: BasisTransform | None = field(default=None, repr=False)


def compile_operation(
    weight: torch.Tensor,
    *,
    threshold: int,
    slice_count: int,
    inner_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    share_count: int = 2,
    seed: bytes | None = None,
    challenge_seed: int | None = None,
    input_basis: BasisTransform | None = None,
    output_basis: BasisTransform | None = None,
    materialize_shares: bool = True,
) -> CompiledOperation:
    if inner_transform is not None and (input_basis is not None or output_basis is not None):
        raise ValueError("inner_transform cannot be combined with client bases")
    if (input_basis is None) != (output_basis is None):
        raise ValueError("inner algebraic protection requires both client bases")
    if input_basis is not None and output_basis is not None:
        input_dimension = (
            input_basis.permutation.numel()
            if isinstance(input_basis, SignedPermutation)
            else input_basis.shape[0]
        )
        output_dimension = (
            output_basis.permutation.numel()
            if isinstance(output_basis, SignedPermutation)
            else output_basis.shape[0]
        )
        if input_dimension != weight.shape[1] or output_dimension != weight.shape[0]:
            raise ValueError("client basis dimensions do not match the weight")
        if isinstance(input_basis, SignedPermutation) != isinstance(
            output_basis, SignedPermutation
        ):
            raise ValueError("client bases must use the same representation")
        if isinstance(input_basis, SignedPermutation) and isinstance(
            output_basis, SignedPermutation
        ):
            weight = output_basis.apply(weight.T).T
            weight = input_basis.apply(weight)
        else:
            assert isinstance(input_basis, torch.Tensor) and isinstance(output_basis, torch.Tensor)
            weight = output_basis @ weight @ input_basis.T
    if inner_transform is not None:
        weight = inner_transform(weight)
    quantized = symmetric_quantize(weight, bits=4)
    if seed is None:
        grid = split_secret(quantized.require_values(), count=share_count, modulus=FIELD_MODULUS)
    else:
        grid = seeded_split_secret(quantized.require_values(), seed=seed, count=share_count)
    shares: dict[int, WeightShare] = {}
    storage: dict[int, StoredWeightShare] = {}
    for index, matrix in enumerate(grid.shares):
        if challenge_seed is None:
            challenge_values = [secrets.randbelow(PRIME_MODULUS) for _ in range(matrix.shape[0])]
            challenge_vector = torch.tensor(challenge_values, dtype=torch.int64)
        else:
            generator = torch.Generator().manual_seed(challenge_seed + index)
            challenge_vector = torch.randint(
                0, PRIME_MODULUS, (matrix.shape[0],), generator=generator, dtype=torch.int64
            )
        challenge = freivalds_challenge(matrix, r=challenge_vector)
        from xurcroh.fortress.arithmetic import seeded_share_seed

        shares[index] = WeightShare(
            weight_index=index,
            matrix=matrix if materialize_shares else None,
            challenge=challenge,
            provider_id=f"weight-share-{index}",
            seed=(
                seeded_share_seed(seed, index)
                if seed is not None and index < share_count - 1
                else None
            ),
        )
        if materialize_shares and shares[index].seed is None:
            payload = matrix.detach().cpu().numpy().tobytes()
            storage[index] = StoredWeightShare(
                shape=tuple(matrix.shape),
                bundle=disperse_shard(payload, threshold=threshold, slice_count=slice_count),
            )
    return CompiledOperation(
        quantized=quantized,
        shares=shares,
        storage=storage,
        input_basis=input_basis,
        output_basis=output_basis,
    )


def materialize_weight_shares(operation: CompiledOperation) -> list[torch.Tensor]:
    from xurcroh.fortress.arithmetic import _seeded_uniform

    matrices: list[torch.Tensor] = []
    for index, share in sorted(operation.shares.items()):
        if share.matrix is not None:
            matrix = share.matrix
        elif share.seed is not None:
            matrix = _seeded_uniform(share.seed, tuple(operation.quantized.require_values().shape))
        elif index == len(operation.shares) - 1:
            total = torch.sum(torch.stack(matrices), dim=0)
            matrix = (
                operation.quantized.require_values().to(torch.int64).remainder(FIELD_MODULUS)
                - total
            ).remainder(FIELD_MODULUS)
        else:
            raise ValueError("non-final weight share has neither a matrix nor a seed")
        matrices.append(matrix)
    return matrices


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


def apply_input_basis(operation: CompiledOperation, hidden: torch.Tensor) -> torch.Tensor:
    basis = operation.input_basis
    if basis is None:
        return hidden
    if isinstance(basis, SignedPermutation):
        return basis.apply(hidden)
    return hidden @ basis.T


def apply_output_basis(operation: CompiledOperation, hidden: torch.Tensor) -> torch.Tensor:
    basis = operation.output_basis
    if basis is None:
        return hidden
    if isinstance(basis, SignedPermutation):
        return basis.inverse(hidden)
    return hidden @ basis
