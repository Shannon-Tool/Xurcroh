from __future__ import annotations

import pytest
import xurcroh_quic


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
