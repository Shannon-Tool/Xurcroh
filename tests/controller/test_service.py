from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from integration.test_stage_sdk import running_stage
from shards.identities import StageIdentities
from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.registry import WorkerRegistry
from xurcroh.controller.service import create_controller_app
from xurcroh.runtime.stage_sessions import ClientStageSession


def test_circuit_relays_real_stage_execution_without_plaintext_logs(
    tiny_llama_path: Path, tmp_path: Path, caplog
) -> None:
    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    with running_stage(compiled.manifest_path, 0, identities.private_keys[0]) as endpoint:
        private = Ed25519PrivateKey.generate()
        registration = Registration(
            worker_id="local-stage-worker",
            endpoint=endpoint,
            models=["tiny"],
            modalities=["text"],
            timestamp=time.time(),
            engine="dense-stage",
        )
        registry = WorkerRegistry()
        registry.register(
            registration, sign_registration(private, registration), private.public_key()
        )
        app = create_controller_app(registry, require_authentication=False)
        with TestClient(app) as client:
            response = client.post(
                "/v1/circuits",
                json={
                    "model": "tiny",
                    "pipeline_parallel": 1,
                    "tensor_parallel": 1,
                    "expert_parallel": 1,
                },
            )
            assert response.status_code == 200
            circuit = response.json()
            assert circuit["stages"][0]["worker"]["worker_id"] == "local-stage-worker"
            assert "return_key" not in response.text
            session = ClientStageSession(0, identities.recipients[0].public_key)
            hidden = torch.randn(2, 16, dtype=torch.float64)
            packet = session.sender.seal("relay-request", "f64", [2, 16], hidden.numpy().tobytes())
            body = {
                "request_id": "relay-request",
                "session_id": session.session_id,
                "session_token": session.session_token,
                "client_public": session.client_public,
                "handshake": base64.b64encode(session.handshake).decode(),
                "stage_index": 0,
                "dtype": "f64",
                "shape": [2, 16],
                "packet": base64.b64encode(packet).decode(),
            }
            with caplog.at_level(logging.INFO):
                relay = client.post(
                    f"/v1/relay/{circuit['circuit_id']}",
                    json=body,
                    headers={"X-Xurcroh-Stage": "0"},
                )
            assert relay.status_code == 200
            result = relay.json()
            assert result["packet"] != body["packet"]
            dtype, shape, payload = session.receiver.open_tensor(
                base64.b64decode(result["packet"], validate=True)
            )
            assert dtype == "f64"
            assert shape == [2, 16]
            assert len(payload) == 2 * 16 * 8
            assert "protected-inference-payload" not in caplog.text
            assert client.get("/metrics").json()["circuits_active"] == 1
            assert client.delete(f"/v1/circuits/{circuit['circuit_id']}").status_code == 204
