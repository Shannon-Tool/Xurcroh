from __future__ import annotations

import pytest
import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.algebra.feedforward import (
    FeedForwardWeights,
    conjugate_feedforward,
    reference_feedforward,
)


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_contragredient_feedforward_matches_reference(activation: str) -> None:
    torch.manual_seed(41)
    d, d_ff = 8, 12
    weights = FeedForwardWeights(
        gate=torch.randn(d_ff, d, dtype=torch.float64),
        up=torch.randn(d_ff, d, dtype=torch.float64),
        down=torch.randn(d, d_ff, dtype=torch.float64),
    )
    basis_in = sample_orthogonal(d, seed=43)
    basis_out = sample_orthogonal(d, seed=47)
    permutation = torch.randperm(d_ff, generator=torch.Generator().manual_seed(53))
    scales = torch.linspace(0.5, 2.0, d_ff, dtype=torch.float64)
    transformed = conjugate_feedforward(
        weights,
        basis_in,
        basis_out,
        permutation,
        scales,
    )
    x = torch.randn(5, d, dtype=torch.float64)
    original = reference_feedforward(x, weights, activation)
    rotated = reference_feedforward(x @ basis_in.T, transformed, activation)
    torch.testing.assert_close(rotated @ basis_out, original, atol=3e-12, rtol=3e-12)


def test_feedforward_rejects_nonpositive_scaling() -> None:
    weights = FeedForwardWeights(
        gate=torch.ones(2, 2),
        up=torch.ones(2, 2),
        down=torch.ones(2, 2),
    )
    with pytest.raises(ValueError, match="positive"):
        conjugate_feedforward(
            weights,
            torch.eye(2),
            torch.eye(2),
            torch.arange(2),
            torch.tensor([1.0, 0.0]),
        )
