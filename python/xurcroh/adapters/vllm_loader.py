from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch

from xurcroh.errors import ModelLoadError
from xurcroh.shards.format import read_shard, validate_shard_path
from xurcroh.shards.key_broker import resolve_stage_identity
from xurcroh.shards.loader import SecureShardLoader
from xurcroh.shards.recipients import open_stage_key, verify_manifest_digest

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
        manifest_value = extra.get("manifest_path")
        if manifest_value is not None:
            manifest_path = Path(manifest_value)
            document = json.loads(manifest_path.read_text())
            if document["format"] != "xurcroh-manifest" or document["version"] != 2:
                raise ModelLoadError("strict vLLM loading requires manifest v2")
            digest = verify_manifest_digest(document)
            stage_index = int(extra["stage_index"])
            stage = document["stages"][stage_index]
            handle = extra.get("stage_identity_handle")
            if not isinstance(handle, str) or not handle:
                raise ModelLoadError("strict vLLM loading requires a recipient identity handle")
            private_key = resolve_stage_identity(handle)
            envelope = json.loads((manifest_path.parent / stage["key_envelope"]).read_text())
            key = open_stage_key(
                envelope,
                private_key,
                manifest_digest=digest,
                checkpoint_fingerprint=document["checkpoint_fingerprint"],
            )
            return SecureShardLoader(
                manifest_path.parent / stage["file"],
                key,
                expected_metadata={
                    "stage_index": stage_index,
                    "stage_count": document["stage_count"],
                    "checkpoint_fingerprint": document["checkpoint_fingerprint"],
                },
            )
        legacy_key = extra.get("stage_key")
        if not isinstance(legacy_key, bytes) or len(legacy_key) != 32:
            raise ModelLoadError(
                "strict vLLM loading requires a 32-byte stage key from the key provider"
            )
        if "shard_path" not in extra:
            raise ModelLoadError("strict vLLM loading requires shard_path")
        return SecureShardLoader(
            extra["shard_path"], legacy_key, expected_metadata=extra.get("expected_metadata")
        )

    def download_model(self, model_config: Any) -> None:
        extra = self.load_config.model_loader_extra_config or {}
        if "shard_path" in extra:
            validate_shard_path(extra["shard_path"])
            return
        if "manifest_path" in extra:
            manifest_path = Path(extra["manifest_path"])
            document = json.loads(manifest_path.read_text())
            stage = document["stages"][int(extra["stage_index"])]
            validate_shard_path(manifest_path.parent / stage["file"])
            return
        raise ModelLoadError("strict vLLM loading does not download plaintext models")

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


def ensure_xurcroh_loader() -> None:
    if register_model_loader is None:
        raise ModelLoadError("vLLM model loader registration is unavailable")
    register_model_loader("xurcroh_sharded")(XurcrohShardedModelLoader)


if register_model_loader is not None:
    ensure_xurcroh_loader()
