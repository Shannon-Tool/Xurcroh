from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignedPermutation:
    permutation: torch.Tensor
    signs: torch.Tensor

    @classmethod
    def generate(cls, dimension: int, *, seed: int) -> SignedPermutation:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(dimension, generator=generator)
        signs = torch.where(torch.rand(dimension, generator=generator) < 0.5, -1.0, 1.0)
        return cls(permutation=permutation, signs=signs)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        permutation = self.permutation.to(values.device)
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        return values[..., permutation] * signs

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        inverse_permutation = torch.argsort(self.permutation.to(values.device))
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        return (values * signs)[..., inverse_permutation]


def adaln(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return hidden * (1 + gamma) + beta + alpha * residual


def block_scalar_adaln(
    hidden: torch.Tensor,
    gamma: float,
    beta: torch.Tensor,
    alpha: float,
    residual: torch.Tensor,
) -> torch.Tensor:
    return hidden * (1 + gamma) + beta + alpha * residual
