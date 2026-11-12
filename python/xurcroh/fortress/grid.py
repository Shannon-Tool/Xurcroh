from __future__ import annotations

import secrets
from dataclasses import dataclass

import torch

from xurcroh.fortress.arithmetic import (
    RING_MODULUS,
    ExactFreivaldsChallenge,
    _modular_matvec,
    verify_freivalds,
)


@dataclass(frozen=True)
class WeightShare:
    weight_index: int
    matrix: torch.Tensor
    challenge: ExactFreivaldsChallenge
    provider_id: str
    replica_provider_id: str | None = None


class RoleExecutionError(ValueError):
    def __init__(
        self,
        activation_index: int,
        weight_index: int,
        provider_id: str,
        reason: str,
    ) -> None:
        self.activation_index = activation_index
        self.weight_index = weight_index
        self.provider_id = provider_id
        self.reason = reason
        super().__init__(
            f"role ({activation_index}, {weight_index}) at {provider_id} failed: {reason}"
        )


def execute_grid(
    activation_shares: list[torch.Tensor],
    weight_shares: dict[int, WeightShare],
    *,
    replica_fraction: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= replica_fraction <= 1.0:
        raise ValueError("replica fraction must be between zero and one")
    if not activation_shares or not weight_shares:
        raise ValueError("share grid requires activation and weight shares")
    results: list[torch.Tensor] = []
    for activation_index, activation in enumerate(activation_shares):
        if activation.dtype != torch.int64:
            raise ValueError("activation shares must use int64")
        for weight_index, share in weight_shares.items():
            if share.weight_index != weight_index or share.matrix.dtype != torch.int64:
                raise ValueError("weight share index or dtype is invalid")
            product = _modular_matvec(share.matrix.T, activation, RING_MODULUS)
            verified_product = _modular_matvec(share.matrix.T, activation, share.challenge.prime)
            if not verify_freivalds(activation, verified_product, share.challenge):
                raise RoleExecutionError(
                    activation_index, weight_index, share.provider_id, "Freivalds check failed"
                )
            if secrets.randbelow(100) < replica_fraction * 100:
                replica = _modular_matvec(share.matrix.T, activation, RING_MODULUS)
                if not torch.equal(product, replica):
                    raise RoleExecutionError(
                        activation_index,
                        weight_index,
                        share.replica_provider_id or share.provider_id,
                        "replica result does not match",
                    )
            results.append(product)
    return torch.sum(torch.stack(results), dim=0).remainder(RING_MODULUS)
