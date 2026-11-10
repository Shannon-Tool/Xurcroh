from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from xurcroh.models.dense import DenseStageRunner, execute_dense_layer, layer_weights_from_dict
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
from xurcroh.shards.format import read_shard
from xurcroh.shards.keys import read_key_envelope


def test_real_two_stage_dense_pipeline_matches_original(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    master_key = bytes(range(32))
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=2,
            master_key=master_key,
            seed=241,
        )
    ).compile()
    original = {
        name: tensor.double()
        for name, tensor in load_file(str(tiny_llama_path / "model.safetensors")).items()
    }
    hidden = torch.randn(5, 16, dtype=torch.float64)
    expected = hidden
    for layer in range(2):
        expected = execute_dense_layer(
            expected,
            layer_weights_from_dict(original, f"model.layers.{layer}", original_weights=True),
            query_heads=4,
            kv_heads=2,
        )

    stages = []
    for index, shard in enumerate(result.stage_shards):
        key = read_key_envelope(result.key_envelopes[index], master_key)
        tensors, metadata = read_shard(shard, key)
        layers = [
            layer_weights_from_dict(tensors, f"model.layers.{layer}")
            for layer in metadata["layer_indices"]
        ]
        stages.append((layers, tensors["stage.input_basis"], tensors.get("stage.transition")))
    actual = DenseStageRunner(stages).run(hidden @ stages[0][1].T)
    torch.testing.assert_close(actual @ stages[-1][1], expected, atol=3e-10, rtol=3e-10)


def test_compiled_pipeline_matches_huggingface_llama_logits(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from transformers import AutoModelForCausalLM

    from xurcroh.client.boundary import ClientBoundary

    master_key = bytes(range(32))
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=2,
            master_key=master_key,
            seed=241,
        )
    ).compile()
    token_ids = torch.tensor([[1, 2, 3, 4]])
    boundary = ClientBoundary(result.boundary_shard, master_key)
    hidden = boundary.embed(token_ids)[0]
    for index, shard in enumerate(result.stage_shards):
        key = read_key_envelope(result.key_envelopes[index], master_key)
        tensors, metadata = read_shard(shard, key)
        layers = [
            layer_weights_from_dict(tensors, f"model.layers.{layer}")
            for layer in metadata["layer_indices"]
        ]
        hidden = DenseStageRunner(
            [(layers, tensors["stage.input_basis"], tensors.get("stage.transition"))]
        ).run(hidden)
    actual = boundary.logits(hidden)
    model = AutoModelForCausalLM.from_pretrained(tiny_llama_path).double().eval()
    with torch.inference_mode():
        expected = model(input_ids=token_ids).logits[0]
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-6)
    boundary.close()
