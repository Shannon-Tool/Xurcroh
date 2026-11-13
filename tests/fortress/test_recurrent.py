from __future__ import annotations

import torch

from xurcroh.fortress.recurrent import (
    DeltaNetBlockWeights,
    FortressDeltaNetCombiner,
    compile_deltanet_operations,
)


def test_deltanet_recurrent_state_stays_exact_against_local_reference() -> None:
    generator = torch.Generator().manual_seed(918)

    def matrix(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64) / 8

    weights = DeltaNetBlockWeights(
        query=matrix(4, 4),
        key=matrix(4, 4),
        value=matrix(4, 4),
        beta=matrix(4, 4),
        decay=matrix(4, 4),
        output=matrix(4, 4),
    )
    operations = compile_deltanet_operations(weights)
    combiner = FortressDeltaNetCombiner(weights, operations)
    hidden = torch.tensor([[0.2, -0.1, 0.3, 0.4], [0.5, 0.2, -0.2, 0.1]], dtype=torch.float64)
    expected = combiner.reference(hidden)
    actual, final_state = combiner.run(hidden)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert final_state.shape == (4, 4)
    assert all(operation.input_basis is not None for operation in operations.values())
