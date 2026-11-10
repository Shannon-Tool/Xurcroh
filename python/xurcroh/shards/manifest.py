from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.format import read_shard
from xurcroh.shards.keys import read_key_envelope


@dataclass(frozen=True)
class ManifestVerification:
    checkpoint_fingerprint: str
    stage_count: int
    tensor_count: int


def _artifact_path(root: Path, name: str) -> Path:
    if Path(name).name != name or not name:
        raise ShardIntegrityError("manifest contains an unsafe artifact path")
    return root / name


def verify_manifest(path: str | Path, master_key: bytes) -> ManifestVerification:
    manifest_path = Path(path)
    try:
        document = json.loads(manifest_path.read_text())
        if document["format"] != "xurcroh-manifest" or document["version"] != 1:
            raise ShardIntegrityError("unsupported shard manifest")
        root = manifest_path.parent
        tensor_count = 0
        fingerprints = set()
        layer_indices = []
        for expected_index, stage in enumerate(document["stages"]):
            stage_path = _artifact_path(root, stage["file"])
            key = read_key_envelope(_artifact_path(root, stage["key_envelope"]), master_key)
            tensors, metadata = read_shard(stage_path, key)
            if metadata["stage_index"] != expected_index:
                raise ShardIntegrityError("non-sequential manifest stage index")
            if metadata["ciphertext_sha256"] != stage["ciphertext_sha256"]:
                raise ShardIntegrityError("manifest ciphertext digest mismatch")
            if metadata["shard_id"] != stage["shard_id"]:
                raise ShardIntegrityError("manifest shard identity mismatch")
            fingerprints.add(metadata["checkpoint_fingerprint"])
            layer_indices.extend(metadata["layer_indices"])
            tensor_count += len(tensors)
        boundary = document["boundary"]
        _, boundary_metadata = read_shard(_artifact_path(root, boundary["file"]), master_key)
        if boundary_metadata["ciphertext_sha256"] != boundary["ciphertext_sha256"]:
            raise ShardIntegrityError("boundary ciphertext digest mismatch")
        fingerprints.add(boundary_metadata["checkpoint_fingerprint"])
        expected_fingerprint = document["checkpoint_fingerprint"]
        if fingerprints != {expected_fingerprint}:
            raise ShardIntegrityError("checkpoint fingerprint mismatch across shards")
        if layer_indices != list(range(document["layer_count"])):
            raise ShardIntegrityError("manifest layer coverage is incomplete")
        return ManifestVerification(
            checkpoint_fingerprint=expected_fingerprint,
            stage_count=len(document["stages"]),
            tensor_count=tensor_count,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ShardIntegrityError):
            raise
        raise ShardIntegrityError(f"invalid manifest {manifest_path}") from error
