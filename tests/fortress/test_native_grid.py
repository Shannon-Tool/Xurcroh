from __future__ import annotations

import pytest
import xurcroh_quic


def test_native_challenge_consumes_packed_integer_matrices() -> None:
    import torch

    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    result = xurcroh_quic.fortress_challenge_values(
        2, 2, weight.contiguous().numpy().tobytes(), [11, 13], 2**32 - 5
    )
    assert result == [50, 74]


def test_native_grid_executes_rust_engine_and_attributes_tampering() -> None:
    share = xurcroh_quic.FortressWeightShare(
        0,
        2,
        2,
        [1, 2, 3, 4],
        [11, 13],
        [50, 74],
        2**32 - 5,
        "provider-0",
        None,
    )
    assert xurcroh_quic.execute_fortress_grid([[5, 7]], [share], 1.0) == [19, 43]
    assert xurcroh_quic.execute_fortress_role([5, 7], share) == [19, 43]
    assert xurcroh_quic.verify_fortress_product([5, 7], [19, 43], [11, 13], [50, 74], 2**32 - 5)
    assert not xurcroh_quic.verify_fortress_product([5, 7], [20, 43], [11, 13], [50, 74], 2**32 - 5)

    tampered = xurcroh_quic.FortressWeightShare(
        0,
        2,
        2,
        [2, 2, 3, 4],
        [11, 13],
        [50, 74],
        2**32 - 5,
        "provider-0",
        None,
    )
    with pytest.raises(ValueError, match="provider-0"):
        xurcroh_quic.execute_fortress_grid([[5, 7]], [tampered], 0.0)
