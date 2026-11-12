from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from shards.identities import StageIdentities
from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
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
        assert metadata["storage_dtype"] == "f64"
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


@pytest.mark.parametrize(
    ("compute_dtype", "shard_dtype", "tolerance"),
    [(torch.float64, None, 1e-7), (torch.float32, torch.bfloat16, 0.05)],
)
def test_qwen_compiler_preserves_original_decoder_outputs(
    tiny_llama_path: Path,
    tmp_path: Path,
    compute_dtype: torch.dtype,
    shard_dtype: torch.dtype | None,
    tolerance: float,
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
    from xurcroh.shards.compiler import XurcrohWeightCompiler

    compiled = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=checkpoint,
            output_dir=tmp_path / "compiled-qwen",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=487,
            compute_dtype=compute_dtype,
            shard_dtype=shard_dtype,
        )
    ).compile()
    boundary = ClientBoundary(compiled.boundary_shard, identities.boundary_key)
    assert boundary.input_basis.dtype == compute_dtype
    assert boundary.output_basis.dtype == compute_dtype
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
        stage_dtype = stage._tensors["model.layers.0.self_attn.q_proj.weight"].dtype
        transformed = stage.execute(boundary.embed(token_ids)[0].to(stage_dtype))
        actual = boundary.recover(transformed)
    torch.testing.assert_close(actual.double(), hidden[0], atol=tolerance, rtol=tolerance)
    stage.close()
    boundary.close()


def test_compiler_reads_original_weights_one_layer_at_a_time(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from xurcroh.shards.compiler import LAYER_PATTERN

    identities = StageIdentities.create(1)
    compiler = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=491,
        )
    )
    read_tensors = compiler.reader.read_tensors
    reads = []

    def tracked_read(names):
        reads.append(list(names))
        return read_tensors(names)

    compiler.reader.read_tensors = tracked_read
    compiler.compile()
    layer_reads = [
        {int(LAYER_PATTERN.search(name).group(1)) for name in names if LAYER_PATTERN.search(name)}
        for names in reads
    ]
    assert layer_reads
    assert all(len(indices) <= 1 for indices in layer_reads)


def test_compiler_seed_defaults_to_non_deterministic_client_secret(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities = StageIdentities.create(1)
    random_result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "random",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
        )
    ).compile()
    zero_result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "zero",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=0,
        )
    ).compile()

    random_boundary, _ = read_shard(random_result.boundary_shard, identities.boundary_key)
    zero_boundary, _ = read_shard(zero_result.boundary_shard, identities.boundary_key)
    assert not torch.equal(random_boundary["basis.input"], zero_boundary["basis.input"])
    assert "seed" not in json.loads(random_result.manifest_path.read_text())
