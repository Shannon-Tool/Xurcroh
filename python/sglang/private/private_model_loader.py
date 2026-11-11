from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.errors import ModelLoadError
from xurcroh.shards.format import read_shard, validate_shard_path
from xurcroh.shards.recipients import open_stage_key, verify_manifest_digest

try:
    from sglang.srt.model_loader.loader import DefaultModelLoader
except ImportError:
    DefaultModelLoader = object  # type: ignore[misc,assignment]


class PrivateModelLoader(DefaultModelLoader):  # type: ignore[misc]
    def _get_all_weights(
        self, model_config: Any, model: torch.nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        extra = self.load_config.model_loader_extra_config or {}
        manifest_path = Path(extra["manifest_path"])
        document = json.loads(manifest_path.read_text())
        if document["format"] != "xurcroh-manifest" or document["version"] != 2:
            raise ModelLoadError("SGLang private loading requires manifest v2")
        digest = verify_manifest_digest(document)
        stage_index = int(extra["stage_index"])
        stage = document["stages"][stage_index]
        identity = extra.get("stage_private_key")
        if not isinstance(identity, bytes) or len(identity) != 32:
            raise ModelLoadError("SGLang private loading requires a recipient identity")
        private_key = X25519PrivateKey.from_private_bytes(identity)
        envelope = json.loads((manifest_path.parent / stage["key_envelope"]).read_text())
        key = open_stage_key(
            envelope,
            private_key,
            manifest_digest=digest,
            checkpoint_fingerprint=document["checkpoint_fingerprint"],
        )
        tensors, metadata = read_shard(manifest_path.parent / stage["file"], key)
        try:
            expected = {
                "stage_index": stage_index,
                "stage_count": document["stage_count"],
                "checkpoint_fingerprint": document["checkpoint_fingerprint"],
            }
            if any(metadata.get(name) != value for name, value in expected.items()):
                raise ModelLoadError("SGLang private shard metadata mismatch")
            yield from tensors.items()
        finally:
            for tensor in tensors.values():
                tensor.zero_()
            tensors.clear()

    def download_model(self, model_config: Any) -> None:
        extra = self.load_config.model_loader_extra_config or {}
        manifest_path = Path(extra["manifest_path"])
        document = json.loads(manifest_path.read_text())
        stage = document["stages"][int(extra["stage_index"])]
        validate_shard_path(manifest_path.parent / stage["file"])
