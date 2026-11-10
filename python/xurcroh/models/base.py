from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: torch.dtype
    shape: tuple[int, ...]
    device: str

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


class StageRunner(Protocol):
    def run(self, hidden: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ModelCapabilities:
    modalities: tuple[str, ...]
    engine: str
    pipeline_parallel: bool = True
    tensor_parallel: bool = True
    expert_parallel: bool = False
    streaming: bool = True
