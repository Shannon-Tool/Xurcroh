from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import torch

from xurcroh.fortress.arithmetic import (
    FIELD_MODULUS,
    ExactFreivaldsChallenge,
    _modular_matmul,
)


@dataclass(frozen=True)
class WeightShare:
    weight_index: int
    matrix: torch.Tensor | None
    challenge: ExactFreivaldsChallenge
    provider_id: str
    replica_provider_id: str | None = None
    seed: bytes | None = field(default=None, repr=False)
    extra_challenges: tuple[ExactFreivaldsChallenge, ...] = ()

    @property
    def challenges(self) -> tuple[ExactFreivaldsChallenge, ...]:
        """Every independent challenge a worker result must satisfy."""

        return (self.challenge, *self.extra_challenges)


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
            if (
                share.weight_index != weight_index
                or share.matrix is None
                or share.matrix.dtype != torch.int64
            ):
                raise ValueError("local execution requires materialized int64 weight shares")
            product = _modular_matmul(activation, share.matrix.T, FIELD_MODULUS)
            if secrets.randbelow(100) < replica_fraction * 100:
                replica = _modular_matmul(activation, share.matrix.T, FIELD_MODULUS)
                if not torch.equal(product, replica):
                    raise RoleExecutionError(
                        activation_index,
                        weight_index,
                        share.replica_provider_id or share.provider_id,
                        "replica result does not match",
                    )
            results.append(product)
    stacked = torch.stack(results)
    combined: list[int | list[int]] = []
    for feature in range(stacked.shape[-1]):
        column = stacked[..., feature]
        if column.ndim == 1:
            combined.append(sum(int(value) for value in column) % FIELD_MODULUS)
        else:
            combined.append(
                [
                    sum(int(value) for value in column[:, index]) % FIELD_MODULUS
                    for index in range(column.shape[1])
                ]
            )
    result = torch.tensor(combined, dtype=torch.int64)
    return result if result.ndim == 1 else result.mT
