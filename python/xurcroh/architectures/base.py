from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from xurcroh.formats import TensorReader


class CanonicalReader(TensorReader):
    def __init__(self, source: TensorReader) -> None:
        super().__init__()
        self.source = source
        self.source_format = source.source_format
        self.architecture = source.architecture
        self.encodings = source.encodings

    @property
    def config(self) -> dict[str, Any]:
        self._ensure_open()
        return self.source.config

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return self.source.metadata

    @property
    def fingerprint(self) -> str:
        self._ensure_open()
        return self.source.fingerprint

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        return {name: tuple(tensor.shape) for name, tensor in self.read_tensors(names).items()}

    def read_operation(self, name: str, columns: slice | None = None) -> torch.Tensor:
        self._ensure_open()
        return (
            self.read_tensors([name])[name] if columns is None else self.read_columns(name, columns)
        )

    def read_columns(self, name: str, columns: slice) -> torch.Tensor:
        return self.read_tensors([name])[name][columns, :]

    def close(self) -> None:
        self.source.close()
        super().close()


@dataclass
class ArchitectureAdapter:
    reader: TensorReader
    name: str
    supported_workflows: tuple[str, ...] = ("text",)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> ArchitectureAdapter:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
