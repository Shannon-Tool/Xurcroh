from __future__ import annotations

import pytest

from xurcroh.models.parallelism import ParallelismConfig, validate_parallelism


def test_sparse_attention_rejects_ring_attention() -> None:
    with pytest.raises(ValueError, match="Ring Attention"):
        validate_parallelism(
            ParallelismConfig(
                ring_degree=2,
                ulysses_degree=1,
                attention_backend="cube_sparse_attn",
            )
        )


def test_usp_and_cfg_parallelism_are_validated() -> None:
    config = ParallelismConfig(
        ring_degree=0,
        ulysses_degree=4,
        pipefusion=True,
        cfg_parallel=True,
        attention_backend="flash",
        cache_dit=True,
    )
    assert validate_parallelism(config) is None
    with pytest.raises(ValueError, match="PipeFusion cannot share"):
        validate_parallelism(
            ParallelismConfig(
                ring_degree=2,
                ulysses_degree=2,
                pipefusion=True,
                attention_backend="flash",
            )
        )
