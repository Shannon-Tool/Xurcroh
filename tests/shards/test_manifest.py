from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
from xurcroh.shards.manifest import verify_manifest


def test_manifest_verification_detects_tampering(tiny_llama_path: Path, tmp_path: Path) -> None:
    master_key = bytes(range(32))
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=2,
            master_key=master_key,
            seed=239,
        )
    ).compile()
    report = verify_manifest(result.manifest_path, master_key)
    assert report.stage_count == 2
    assert report.tensor_count > 0

    result.stage_shards[0].write_bytes(result.stage_shards[0].read_bytes()[:-1] + b"x")
    with pytest.raises(ShardIntegrityError):
        verify_manifest(result.manifest_path, master_key)
