from __future__ import annotations

import json
from pathlib import Path

import pytest

from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
from xurcroh.shards.format import read_shard
from xurcroh.shards.keys import read_key_envelope


def test_compiler_partitions_dense_checkpoint_and_separates_boundary(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "compiled"
    master_key = bytes(range(32))
    config = CompilationConfig(
        checkpoint=tiny_llama_path,
        output_dir=output,
        stage_count=2,
        master_key=master_key,
        seed=211,
    )
    result = XurcrohWeightCompiler(config).compile()
    assert len(result.stage_shards) == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["stage_count"] == 2
    assert manifest["boundary"]["tied_embeddings"] is False

    boundary, _ = read_shard(result.boundary_shard, master_key)
    assert "embedding.weight" in boundary
    assert "lm_head.weight" in boundary
    assert "basis.input" in boundary
    assert "basis.output" in boundary

    original_config = json.loads((tiny_llama_path / "config.json").read_text())
    layer_count = original_config["num_hidden_layers"]
    seen_layers: list[int] = []
    for stage_index, shard_path in enumerate(result.stage_shards):
        key = read_key_envelope(result.key_envelopes[stage_index], master_key)
        tensors, metadata = read_shard(shard_path, key)
        assert metadata["stage_index"] == stage_index
        assert not any("embed_tokens" in name or "lm_head" in name for name in tensors)
        seen_layers.extend(metadata["layer_indices"])
        assert "stage.input_basis" in tensors
        assert "stage.output_basis" in tensors
        if stage_index + 1 < len(result.stage_shards):
            assert "stage.transition" in tensors
    assert seen_layers == list(range(layer_count))


def test_compiler_rejects_non_divisible_stage_count(tiny_llama_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage_count"):
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "bad",
            stage_count=0,
            master_key=bytes(32),
            seed=1,
        )
