from __future__ import annotations

from pathlib import Path

import pytest

from shards.identities import StageIdentities
from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.manifest import verify_manifest


def test_manifest_verification_detects_tampering(tiny_llama_path: Path, tmp_path: Path) -> None:
    identities = StageIdentities.create(2)
    result = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=239)
    private_keys = dict(enumerate(identities.private_keys))
    report = verify_manifest(
        result.manifest_path,
        identities.boundary_key,
        stage_private_keys=private_keys,
    )
    assert report.stage_count == 2
    assert report.tensor_count > 0

    result.stage_shards[0].write_bytes(result.stage_shards[0].read_bytes()[:-1] + b"x")
    with pytest.raises(ShardIntegrityError):
        verify_manifest(
            result.manifest_path,
            identities.boundary_key,
            stage_private_keys=private_keys,
        )
