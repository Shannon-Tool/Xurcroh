from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save as save_tensors

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.format import read_shard, validate_shard_path, write_shard


def test_encrypted_shard_round_trip_and_authentication(tmp_path: Path) -> None:
    path = tmp_path / "stage-0.xurcroh"
    key = bytes(range(32))
    tensors = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    metadata = write_shard(path, tensors, key, {"model": "tiny", "stage_index": 0})
    assert path.read_bytes() != save_tensors(tensors)
    loaded, loaded_metadata = read_shard(path, key)
    torch.testing.assert_close(loaded["weight"], tensors["weight"])
    assert loaded_metadata["shard_id"] == metadata["shard_id"]
    assert loaded_metadata["ciphertext_sha256"] == metadata["ciphertext_sha256"]

    with pytest.raises(ShardIntegrityError):
        read_shard(path, bytes(reversed(range(32))))

    tampered = bytearray(path.read_bytes())
    tampered[-1] ^= 1
    path.write_bytes(tampered)
    with pytest.raises(ShardIntegrityError):
        read_shard(path, key)


def test_missing_truncated_and_plaintext_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ShardIntegrityError, match="missing"):
        read_shard(tmp_path / "missing.xurcroh", bytes(32))
    with pytest.raises(ShardIntegrityError):
        read_shard(tmp_path / "missing.xurcroh", bytes(32))
    plaintext = tmp_path / "model.safetensors"
    plaintext.write_bytes(b"unprotected")
    with pytest.raises(ShardIntegrityError, match="strict"):
        validate_shard_path(plaintext)


def test_truncated_shard_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "stage.xurcroh"
    write_shard(path, {"x": torch.ones(2)}, bytes(32), {"stage": 0})
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(ShardIntegrityError):
        read_shard(path, bytes(32))
