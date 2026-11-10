from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeedForwardWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


def reference_feedforward(
    hidden: torch.Tensor,
    weights: FeedForwardWeights,
    activation: Literal["silu", "gelu"],
) -> torch.Tensor:
    gate = hidden @ weights.gate.T
    up = hidden @ weights.up.T
    nonlinear = F.silu(gate) if activation == "silu" else F.gelu(gate)
    return (nonlinear * up) @ weights.down.T


def conjugate_feedforward(
    weights: FeedForwardWeights,
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    permutation: torch.Tensor,
    scales: torch.Tensor,
) -> FeedForwardWeights:
    intermediate = weights.gate.shape[0]
    if permutation.shape != (intermediate,) or not torch.equal(
        torch.sort(permutation.cpu()).values, torch.arange(intermediate)
    ):
        raise ValueError("feed-forward permutation must contain every intermediate index once")
    if scales.shape != (intermediate,) or torch.any(scales <= 0):
        raise ValueError("feed-forward contragredient scales must be positive")
    permutation = permutation.to(device=weights.gate.device)
    scales = scales.to(device=weights.gate.device, dtype=weights.gate.dtype)
    gate = weights.gate[permutation] @ input_basis.T
    up = (weights.up[permutation] * scales[:, None]) @ input_basis.T
    down = weights.down[:, permutation] * scales.reciprocal()[None, :]
    down = output_basis @ down
    return FeedForwardWeights(gate=gate, up=up, down=down)
