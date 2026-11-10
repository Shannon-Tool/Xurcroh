from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from integration.test_stage_sdk import running_stage
from shards.identities import StageIdentities
from shards.test_moe_compiler import make_tensors
from Xurcroh import Xurcroh
from xurcroh.algebra.orthogonal import sample_orthogonal
from xurcroh.models.dense import DenseStageRunner
from xurcroh.models.moe import moe_layer_weights_from_dict
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler


def test_moe_stage_matches_original_attention_router_and_experts(
    tiny_llama_path: Path,
) -> None:
    identities = StageIdentities.create(1)
    compiler = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tiny_llama_path / "unused",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=229,
        )
    )
    original = make_tensors()
    basis = sample_orthogonal(16, seed=235, dtype=torch.float64)
    transformed = compiler._compile_layer(original, "model.layers.0", basis, 233)
    prefix = "model.layers.0"
    original_layer = moe_layer_weights_from_dict(original, prefix, original_weights=True, top_k=2)
    transformed_layer = moe_layer_weights_from_dict(transformed, prefix, top_k=2)
    hidden = torch.randn(5, 16, dtype=torch.float64)
    expected = DenseStageRunner([([original_layer], basis, None)]).run(hidden)
    actual = DenseStageRunner([([transformed_layer], basis, None)]).run(hidden @ basis.T)
    torch.testing.assert_close(actual @ basis, expected, atol=3e-10, rtol=3e-10)


def test_compiled_moe_executes_through_remote_stage(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "moe-checkpoint"
    shutil.copytree(tiny_llama_path, checkpoint)
    tensors = dict(load_file(str(checkpoint / "model.safetensors")))
    moe = make_tensors()
    prefix = "model.layers.0.mlp"
    for name in list(tensors):
        if name.startswith(prefix + "."):
            del tensors[name]
    for name, tensor in moe.items():
        if name.startswith(prefix + ".") or "mlp.gate.weight" in name:
            tensors[name] = tensor.contiguous()
    save_file(tensors, str(checkpoint / "model.safetensors"))
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config.update({"num_experts": 3, "num_experts_per_tok": 2, "intermediate_size": 12})
    config_path.write_text(json.dumps(config))
    identities = StageIdentities.create(1)
    compiled = identities.compile(checkpoint, tmp_path / "compiled-moe", seed=241)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    with (
        running_stage(compiled.manifest_path, 0, identities.private_keys[0]) as endpoint,
        Xurcroh(
            cluster={"name": "moe-stage", "endpoint": endpoint},
            model_path={
                "tiny-moe": {
                    "path": str(checkpoint),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
            security={"mode": "development"},
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny-moe",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0,
        )
    assert response.execution.engine == "dense-stages"
    assert response.usage.completion_tokens == 1
