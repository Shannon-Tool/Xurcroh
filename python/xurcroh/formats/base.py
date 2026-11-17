from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from xurcroh.errors import ModelLoadError

CheckpointFormat = Literal["safetensors", "gguf", "exl2", "exl3"]


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    encoding: str
    source: str
    offset: int | None = None


@dataclass(frozen=True)
class ModelSupportReport:
    model: str
    checkpoint: str
    source_format: str
    architecture: str
    encodings: tuple[str, ...]
    supported: bool
    issues: tuple[str, ...]
    required_adapter: str | None


class TensorReader(ABC):
    source_format: CheckpointFormat
    architecture: str
    encodings: tuple[str, ...]

    def __init__(self) -> None:
        self._closed = False

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def fingerprint(self) -> str: ...

    @abstractmethod
    def tensor_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]: ...

    @abstractmethod
    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]: ...

    def read_operation(self, name: str, columns: slice | None = None) -> torch.Tensor:
        self._ensure_open()
        tensor = self.read_tensors([name])[name]
        if tensor.ndim != 2:
            raise ModelLoadError(f"checkpoint operation is not a matrix: {name}")
        return tensor if columns is None else tensor[columns, :]

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelLoadError("checkpoint reader is closed")

    def __enter__(self) -> TensorReader:
        self._ensure_open()
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def fingerprint_source(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
