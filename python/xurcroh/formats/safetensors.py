from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from xurcroh.errors import ModelLoadError
from xurcroh.formats.base import CheckpointFormat, TensorReader, fingerprint_source


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelLoadError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except ModelLoadError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ModelLoadError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ModelLoadError(f"{description} must contain a JSON object: {path}")
    return value


class SafetensorsReader(TensorReader):
    source_format: CheckpointFormat = "safetensors"

    def __init__(self, root: Path) -> None:
        super().__init__()
        if not root.is_dir():
            raise ModelLoadError(f"checkpoint directory does not exist: {root}")
        self.root = root
        self._config = _read_json(root / "config.json", "checkpoint config")
        self.architecture = str(self._config.get("model_type", "unknown"))
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            index = _read_json(index_path, "checkpoint index")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not all(
                isinstance(key, str) and isinstance(filename, str)
                for key, filename in weight_map.items()
            ):
                raise ModelLoadError("checkpoint index has an invalid weight_map")
            self.weight_map = weight_map
        elif (root / "model.safetensors").is_file():
            self.weight_map = {}
            with safe_open(root / "model.safetensors", framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118
                    if name in self.weight_map:
                        raise ModelLoadError(f"duplicate checkpoint tensor: {name}")
                    self.weight_map[name] = "model.safetensors"
        else:
            raise ModelLoadError("strict compiler requires safetensors checkpoint files")
        self._files = [root / "config.json"]
        if index_path.is_file():
            self._files.append(index_path)
        self._files.extend(root / filename for filename in sorted(set(self.weight_map.values())))
        self.encodings = self._read_encodings()

    @property
    def config(self) -> dict[str, Any]:
        self._ensure_open()
        return self._config

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return {"format": self.source_format, "architecture": self.architecture}

    @property
    def fingerprint(self) -> str:
        self._ensure_open()
        return fingerprint_source(self.root, self._files)

    def _read_encodings(self) -> tuple[str, ...]:
        encodings: set[str] = set()
        for filename in sorted(set(self.weight_map.values())):
            with safe_open(self.root / filename, framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118
                    encodings.add(str(handle.get_slice(name).get_dtype()))
        return tuple(sorted(encodings))

    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(self.weight_map)

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        grouped = self._group(names)
        shapes: dict[str, tuple[int, ...]] = {}
        for filename, requested in grouped.items():
            with safe_open(self.root / filename, framework="pt") as handle:
                for name in requested:
                    shapes[name] = tuple(handle.get_slice(name).get_shape())
        return shapes

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        grouped = self._group(names)
        tensors: dict[str, torch.Tensor] = {}
        for filename, requested in grouped.items():
            with safe_open(self.root / filename, framework="pt") as handle:
                for name in requested:
                    tensors[name] = handle.get_tensor(name)
        return tensors

    def _group(self, names: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for name in names:
            if name not in self.weight_map:
                raise ModelLoadError(f"checkpoint tensor is missing: {name}")
            grouped.setdefault(self.weight_map[name], []).append(name)
        return grouped
