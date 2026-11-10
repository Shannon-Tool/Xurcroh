from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismConfig:
    ring_degree: int
    ulysses_degree: int
    attention_backend: str
    pipefusion: bool = False
    cfg_parallel: bool = False
    cache_dit: bool = False


def validate_parallelism(config: ParallelismConfig) -> None:
    if config.ring_degree < 0 or config.ulysses_degree < 1:
        raise ValueError("invalid sequence-parallel degrees")
    sparse = {"cube_sparse_attn", "subblock_sparse_attn"}
    if config.ring_degree > 1 and config.attention_backend in sparse:
        raise ValueError("Ring Attention is incompatible with Cube/SubBlock sparse attention")
    if config.ring_degree > 0 and config.ulysses_degree > 1 and config.pipefusion:
        raise ValueError("PipeFusion cannot share the same sequence dimension with Ring/Ulysses")
