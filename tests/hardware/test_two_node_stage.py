from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
import torch

from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
from xurcroh.kem import HybridKemKey
from xurcroh.models.dense import execute_dense_layer_cached, layer_weights_from_dict
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.stage import DenseStageEngine
from xurcroh.stage_worker import DenseStage

pytestmark = pytest.mark.hardware


def test_two_node_outputs_match_fp64_reference() -> None:
    deployment_path = os.environ.get("XURCROH_TWO_NODE_DEPLOYMENT")
    if deployment_path is None:
        pytest.skip("XURCROH_TWO_NODE_DEPLOYMENT must identify a live two-worker deployment")
    deployment = json.loads(Path(deployment_path).read_text())
    os.environ["SSL_CERT_FILE"] = str(Path(deployment["ca_file"]).resolve())
    with open(deployment["boundary_key_file"]) as handle:
        os.environ["XURCROH_CLIENT_BOUNDARY_KEY"] = handle.read().strip()
    manifest = Path(deployment["manifest"])
    manifest_document = json.loads(manifest.read_text())
    checkpoint_config = json.loads((Path(deployment["checkpoint"]) / "config.json").read_text())
    workers = [
        WorkerConfig(name=item["name"], endpoint=item["endpoint"]) for item in deployment["workers"]
    ]
    engine = DenseStageEngine(
        deployment["model"],
        ModelDefinition(path=deployment["checkpoint"], manifest_path=str(manifest), engine="vllm"),
        workers,
        SecurityConfig(mode="strict"),
    )
    captured: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
        index: [] for index in range(len(workers))
    }
    execute = engine._execute_stage

    def capture(index, hidden, request_id, session, timeout, circuit_id=None):
        result = execute(index, hidden, request_id, session, timeout, circuit_id)
        captured[index].append((hidden.detach().cpu().clone(), result.detach().cpu().clone()))
        return result

    engine._execute_stage = capture  # type: ignore[method-assign]
    request = CompletionRequest.validated(
        model=deployment["model"],
        messages=[{"role": "user", "content": "Reply with a short technical sentence."}],
        temperature=0,
        max_tokens=2,
        request_id="two-node-numerics",
    )
    try:
        list(engine.generate(request, cancelled=threading.Event(), timeout=300))
    finally:
        engine.close()

    errors: dict[str, list[float]] = {}
    try:
        for index in range(len(workers)):
            identity = manifest.parent / "identities" / f"stage-{index}.key"
            private_key = HybridKemKey.from_bytes(identity.read_bytes())
            shard = DenseStage(manifest, index, private_key)
            stage = manifest_document["stages"][index]
            rope_theta = float(
                checkpoint_config.get(
                    "rope_theta",
                    checkpoint_config.get("rope_parameters", {}).get("rope_theta", 10_000),
                )
            )
            stage_errors = []
            errors[f"stage-{index}"] = stage_errors
            weights = {}
            pasts = {}
            try:
                inputs = {
                    token_index: actual_input.double()
                    for token_index, (actual_input, _) in enumerate(captured[index])
                }
                pasts = {token_index: None for token_index in inputs}
                for layer_index in stage["layer_indices"]:
                    prefix = f"model.layers.{layer_index}"
                    weights = {
                        name: value.double()
                        for name, value in shard._tensors.items()
                        if name.startswith(prefix + ".")
                    }
                    layer = layer_weights_from_dict(weights, prefix)
                    next_inputs = {}
                    next_pasts = {}
                    for token_index, current_input in inputs.items():
                        expected, past = execute_dense_layer_cached(
                            current_input,
                            layer,
                            query_heads=stage["query_heads"],
                            kv_heads=stage["kv_heads"],
                            past=None if token_index == 0 else pasts[0],
                            interleaved_rope=stage["interleaved_rope"],
                            rope_base=rope_theta,
                        )
                        next_inputs[token_index] = expected
                        next_pasts[token_index] = past
                    for past in pasts.values():
                        if past is not None:
                            for value in past:
                                value.zero_()
                    inputs = next_inputs
                    pasts = next_pasts
                    for value in weights.values():
                        value.zero_()
                for token_index, (_, actual_output) in enumerate(captured[index]):
                    expected = inputs[token_index]
                    error = (actual_output.double() - expected).abs().max().item()
                    stage_errors.append(error)
                    torch.testing.assert_close(
                        actual_output.double(),
                        expected,
                        atol=0.05,
                        rtol=0.05,
                        msg=lambda message, stage_index=index, token=token_index: (
                            f"stage {stage_index}, token {token} differs from reference: {message}"
                        ),
                    )
            finally:
                shard.close()
                for past in pasts.values():
                    if past is not None:
                        for value in past:
                            value.zero_()
                for value in weights.values():
                    value.zero_()
    except Exception as error:
        report_path = os.environ.get("XURCROH_HARDWARE_REPORT")
        if report_path:
            Path(report_path).write_text(
                json.dumps(
                    {
                        "test": "two-node-vllm-vs-fp64-reference",
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "max_abs_errors": errors,
                        "atol": 0.05,
                        "rtol": 0.05,
                    },
                    indent=2,
                )
                + "\n"
            )
        raise

    report_path = os.environ.get("XURCROH_HARDWARE_REPORT")
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                {
                    "test": "two-node-vllm-vs-fp64-reference",
                    "status": "passed",
                    "revision": deployment["revision"],
                    "max_abs_errors": errors,
                    "atol": 0.05,
                    "rtol": 0.05,
                    "hardware_validated": True,
                    "production_accepted": False,
                },
                indent=2,
            )
            + "\n"
        )
