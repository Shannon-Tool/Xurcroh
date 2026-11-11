from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.stage import DenseStageEngine
from xurcroh.stage_worker import DenseStage

pytestmark = pytest.mark.hardware


def test_two_node_outputs_match_bf16_reference() -> None:
    deployment_path = os.environ.get("XURCROH_TWO_NODE_DEPLOYMENT")
    if deployment_path is None:
        pytest.skip("XURCROH_TWO_NODE_DEPLOYMENT must identify a live two-worker deployment")
    deployment = json.loads(Path(deployment_path).read_text())
    os.environ["SSL_CERT_FILE"] = str(Path(deployment["ca_file"]).resolve())
    with open(deployment["boundary_key_file"]) as handle:
        os.environ["XURCROH_CLIENT_BOUNDARY_KEY"] = handle.read().strip()
    manifest = Path(deployment["manifest"])
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
            private_key = X25519PrivateKey.from_private_bytes(identity.read_bytes())
            reference = DenseStage(manifest, index, private_key)
            stage_errors = []
            reference_dtype = next(
                value.dtype
                for name, value in reference._tensors.items()
                if name.endswith("self_attn.q_proj.weight")
            )
            try:
                for token_index, (actual_input, actual_output) in enumerate(captured[index]):
                    expected = reference.execute(
                        actual_input.to(reference_dtype), f"two-node-reference-{index}"
                    )
                    error = (actual_output.double() - expected.double()).abs().max().item()
                    stage_errors.append(error)
                    torch.testing.assert_close(
                        actual_output.double(),
                        expected.double(),
                        atol=0.05,
                        rtol=0.05,
                        msg=lambda message, stage=index, token=token_index: (
                            f"stage {stage}, token {token} differs from reference: {message}"
                        ),
                    )
                    reference.runner.drop_request(f"two-node-reference-{index}")
            finally:
                reference.close()
            errors[f"stage-{index}"] = stage_errors
    except Exception as error:
        report_path = os.environ.get("XURCROH_HARDWARE_REPORT")
        if report_path:
            Path(report_path).write_text(
                json.dumps(
                    {
                        "test": "two-node-vllm-vs-bf16-reference",
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
                    "test": "two-node-vllm-vs-bf16-reference",
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
