from __future__ import annotations

from collections.abc import Generator
from typing import Any

import torch

from xurcroh.errors import ModelLoadError
from xurcroh.shards.format import read_shard, validate_shard_path
from xurcroh.shards.loader import SecureShardLoader

try:
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
except ImportError:
    BaseModelLoader = object
    register_model_loader = None


class XurcrohShardedModelLoader(BaseModelLoader):  # type: ignore[misc]
    def __init__(self, load_config: Any) -> None:
        if BaseModelLoader is not object:
            super().__init__(load_config)
        self.load_config = load_config

    def _secure_loader(self) -> SecureShardLoader:
        extra = self.load_config.model_loader_extra_config or {}
        key = extra.get("stage_key")
        if not isinstance(key, bytes) or len(key) != 32:
            raise ModelLoadError(
                "strict vLLM loading requires a 32-byte stage key from the key provider"
            )
        if "shard_path" not in extra:
            raise ModelLoadError("strict vLLM loading requires shard_path")
        return SecureShardLoader(
            extra["shard_path"], key, expected_metadata=extra.get("expected_metadata")
        )

    def download_model(self, model_config: Any) -> None:
        extra = self.load_config.model_loader_extra_config or {}
        if "shard_path" not in extra:
            raise ModelLoadError("strict vLLM loading does not download plaintext models")
        validate_shard_path(extra["shard_path"])

    def load_weights(self, model: torch.nn.Module, model_config: Any) -> None:
        self._secure_loader().load_into(model)

    def get_all_weights(
        self, model_config: Any, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        loader = self._secure_loader()
        tensors, _ = read_shard(loader.path, loader.key, expected_metadata=loader.expected_metadata)
        try:
            yield from tensors.items()
        finally:
            for tensor in tensors.values():
                tensor.zero_()
            tensors.clear()


if register_model_loader is not None:
    register_model_loader("xurcroh_sharded")(XurcrohShardedModelLoader)
