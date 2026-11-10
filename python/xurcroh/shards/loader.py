from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xurcroh.errors import ModelLoadError
from xurcroh.shards.format import read_shard
from xurcroh.shards.memory import secure_zero

__all__ = ["SecureShardLoader", "secure_zero"]


class SecureShardLoader:
    def __init__(
        self,
        path: str | Path,
        key: bytes,
        *,
        expected_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.key = key
        self.expected_metadata = expected_metadata

    def load_into(self, model: torch.nn.Module) -> None:
        tensors, _ = read_shard(self.path, self.key, expected_metadata=self.expected_metadata)
        try:
            load_weights = getattr(model, "load_weights", None)
            if callable(load_weights):
                load_weights(tensors.items())
            else:
                model.load_state_dict(tensors, strict=True)
        except Exception as error:
            raise ModelLoadError(f"failed to load secure shard {self.path}: {error}") from error
        finally:
            for tensor in tensors.values():
                tensor.zero_()
            tensors.clear()
