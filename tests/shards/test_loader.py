from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.format import write_shard
from xurcroh.shards.loader import SecureShardLoader, secure_zero


def test_secure_loader_loads_only_expected_stage(tmp_path: Path) -> None:
    path = tmp_path / "stage.xurcroh"
    write_shard(path, {"weight": torch.ones(2, 2)}, bytes(32), {"stage_index": 2})
    module = torch.nn.Linear(2, 2, bias=False)
    loader = SecureShardLoader(path, bytes(32), expected_metadata={"stage_index": 2})
    loader.load_into(module)
    torch.testing.assert_close(module.weight, torch.ones(2, 2))
    with pytest.raises(ShardIntegrityError, match="metadata"):
        SecureShardLoader(path, bytes(32), expected_metadata={"stage_index": 1}).load_into(module)


def test_secure_zero_overwrites_mutable_buffer() -> None:
    buffer = bytearray(b"sensitive")
    secure_zero(buffer)
    assert buffer == bytes(len(buffer))
