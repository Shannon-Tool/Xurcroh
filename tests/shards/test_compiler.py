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
        assert metadata["rope_theta"] == original_config.get(
            "rope_theta", original_config.get("rope_parameters", {}).get("rope_theta", 10_000)
        )
        assert not any("embed_tokens" in name or "lm_head" in name for name in tensors)
        seen_layers.extend(metadata["layer_indices"])
        assert not any("basis" in name or "transition" in name for name in tensors)
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


def test_qwen_compiler_preserves_original_decoder_outputs(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from xurcroh.client.boundary import ClientBoundary
    from xurcroh.stage_worker import DenseStage

    checkpoint = tmp_path / "qwen"
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        head_dim=4,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        tie_word_embeddings=False,
        attention_bias=False,
    )
    original = Qwen3ForCausalLM(config).double().eval()
    original.save_pretrained(checkpoint, safe_serialization=True)
    AutoTokenizer.from_pretrained(tiny_llama_path).save_pretrained(checkpoint)
    identities = StageIdentities.create(1)
    compiled = identities.compile(checkpoint, tmp_path / "compiled-qwen", seed=487)
    boundary = ClientBoundary(compiled.boundary_shard, identities.boundary_key)
    stage = DenseStage(compiled.manifest_path, 0, identities.private_keys[0])
    token_ids = torch.tensor([[2, 3, 2]])
    hidden = original.model.embed_tokens(token_ids)
    positions = torch.arange(token_ids.shape[1]).unsqueeze(0)
    mask = torch.full(
        (1, 1, token_ids.shape[1], token_ids.shape[1]),
        torch.finfo(torch.float64).min,
        dtype=torch.float64,
    ).triu(1)
    with torch.inference_mode():
        for layer in original.model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                position_embeddings=original.model.rotary_emb(hidden, positions),
            )
        transformed = stage.execute(boundary.embed(token_ids)[0])
        actual = boundary.recover(transformed)
    torch.testing.assert_close(actual, hidden[0], atol=1e-7, rtol=1e-7)
    stage.close()
    boundary.close()
