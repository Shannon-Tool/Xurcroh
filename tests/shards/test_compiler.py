from __future__ import annotations

import json
from pathlib import Path

import pytest

from shards.identities import StageIdentities
from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.compiler import CompilationConfig
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key


def test_compiler_partitions_dense_checkpoint_and_separates_boundary(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "compiled"
    identities = StageIdentities.create(2)
    result = identities.compile(tiny_llama_path, output, seed=211)
    assert len(result.stage_shards) == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["version"] == 2
    assert manifest["stage_count"] == 2
    assert manifest["boundary"]["tied_embeddings"] is False

    boundary, _ = read_shard(result.boundary_shard, identities.boundary_key)
    assert "embedding.weight" in boundary
    assert "lm_head.weight" in boundary
    assert "basis.input" in boundary
    assert "basis.output" in boundary

    original_config = json.loads((tiny_llama_path / "config.json").read_text())
    layer_count = original_config["num_hidden_layers"]
    seen_layers: list[int] = []
    for stage_index, shard_path in enumerate(result.stage_shards):
        envelope = json.loads(result.key_envelopes[stage_index].read_text())
        key = open_stage_key(
            envelope,
            identities.private_keys[stage_index],
            manifest_digest=manifest["digest"],
            checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
        )
        tensors, metadata = read_shard(shard_path, key)
        assert metadata["stage_index"] == stage_index
        assert not any("embed_tokens" in name or "lm_head" in name for name in tensors)
        seen_layers.extend(metadata["layer_indices"])
        assert "stage.input_basis" in tensors
        assert "stage.output_basis" in tensors
        if stage_index + 1 < len(result.stage_shards):
            assert "stage.transition" in tensors
    assert seen_layers == list(range(layer_count))

    with pytest.raises(ShardIntegrityError):
        read_shard(result.boundary_shard, bytes(32))


def test_compiler_rejects_non_divisible_stage_count(tiny_llama_path: Path, tmp_path: Path) -> None:
    identities = StageIdentities.create(1)
    with pytest.raises(ValueError, match="stage_count"):
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "bad",
            stage_count=0,
            boundary_key=identities.boundary_key,
            recipients=(),
            seed=1,
        )
