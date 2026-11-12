from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.service import WorkerEnrollment, create_controller_app


def registration(private: Ed25519PrivateKey, endpoint: str) -> Registration:
    return Registration(
        worker_id="worker-a",
        endpoint=endpoint,
        models=["tiny"],
        modalities=["text"],
        timestamp=time.time(),
        engine="dense-stage",
    )


def register(client: TestClient, private: Ed25519PrivateKey, endpoint: str):
    document = registration(private, endpoint)
    return client.post(
        "/v1/workers/register",
        json={
            "registration": document.model_dump(mode="json"),
            "signature": base64.b64encode(sign_registration(private, document)).decode(),
            "public_key": base64.b64encode(private.public_key().public_bytes_raw()).decode(),
        },
    )


def test_controller_requires_enrollment_and_client_authentication() -> None:
    private = Ed25519PrivateKey.generate()
    endpoint = "https://worker-a.invalid"
    app = create_controller_app(
        enrollments={"worker-a": WorkerEnrollment(private.public_key(), endpoint)},
        client_tokens={"secret": "client-a"},
    )
    with TestClient(app) as client:
        assert register(client, private, endpoint).status_code == 200
        assert client.post("/v1/circuits", json={"model": "tiny"}).status_code == 403
        authorized = {"Authorization": "Bearer secret"}
        assert client.get("/v1/workers", headers=authorized).status_code == 200
        response = client.post("/v1/circuits", json={"model": "tiny"}, headers=authorized)
        assert response.status_code == 200
        assert (
            client.post(
                "/v1/circuits", json={"model": "tiny"}, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/v1/relay/{response.json()['circuit_id']}",
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 403
        )


def test_controller_rejects_changed_worker_endpoint() -> None:
    private = Ed25519PrivateKey.generate()
    app = create_controller_app(
        enrollments={
            "worker-a": WorkerEnrollment(private.public_key(), "https://worker-a.invalid")
        },
        client_tokens={"secret": "client-a"},
    )
    with TestClient(app) as client:
        assert register(client, private, "https://attacker.invalid").status_code == 403
