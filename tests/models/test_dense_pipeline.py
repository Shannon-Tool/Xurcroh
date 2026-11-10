from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from shards.identities import StageIdentities
from xurcroh.models.dense import DenseStageRunner, execute_dense_layer, layer_weights_from_dict
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key


def _open_stage(result, identities: StageIdentities, index: int):
    manifest = json.loads(result.manifest_path.read_text())
    envelope = json.loads(result.key_envelopes[index].read_text())
    key = open_stage_key(
        envelope,
        identities.private_keys[index],
        manifest_digest=manifest["digest"],
        checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
    )
    return read_shard(result.stage_shards[index], key)


def test_real_two_stage_dense_pipeline_matches_original(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities = StageIdentities.create(2)
    result = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
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
    for index in range(2):
        tensors, metadata = _open_stage(result, identities, index)
        layers = [
            layer_weights_from_dict(tensors, f"model.layers.{layer}")
            for layer in metadata["layer_indices"]
        ]
        stages.append(layers)
    boundary, _ = read_shard(result.boundary_shard, identities.boundary_key)
    actual = hidden @ boundary["basis.input"].T
    for index, layers in enumerate(stages):
        actual = DenseStageRunner([layers]).run(actual)
        if index + 1 < len(stages):
            actual = (
                actual @ boundary["basis.stages"][index] @ boundary["basis.stages"][index + 1].T
            )
    torch.testing.assert_close(actual @ boundary["basis.output"], expected, atol=3e-10, rtol=3e-10)


def test_compiled_pipeline_matches_huggingface_llama_logits(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from transformers import AutoModelForCausalLM

    from xurcroh.client.boundary import ClientBoundary

    identities = StageIdentities.create(2)
    result = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    token_ids = torch.tensor([[1, 2, 3, 4]])
    boundary = ClientBoundary(result.boundary_shard, identities.boundary_key)
    hidden = boundary.embed(token_ids)[0]
    for index in range(2):
        tensors, metadata = _open_stage(result, identities, index)
        layers = [
            layer_weights_from_dict(tensors, f"model.layers.{layer}")
            for layer in metadata["layer_indices"]
        ]
        hidden = DenseStageRunner([layers]).run(hidden)
        if index + 1 < 2:
            hidden = boundary.transition(hidden, index)
    actual = boundary.logits(hidden)
    model = AutoModelForCausalLM.from_pretrained(tiny_llama_path).double().eval()
    with torch.inference_mode():
        expected = model(input_ids=token_ids).logits[0]
    torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-6)
    boundary.close()
