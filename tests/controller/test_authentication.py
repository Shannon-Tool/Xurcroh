from __future__ import annotations

import base64
import time

from fastapi.testclient import TestClient

from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.service import WorkerEnrollment, create_controller_app
from xurcroh.signatures import HybridSigningKey


def registration(private: HybridSigningKey, endpoint: str) -> Registration:
    return Registration(
        worker_id="worker-a",
        endpoint=endpoint,
        models=["tiny"],
        modalities=["text"],
        timestamp=time.time(),
        engine="dense-stage",
    )


def register(client: TestClient, private: HybridSigningKey, endpoint: str):
    document = registration(private, endpoint)
    return client.post(
        "/v1/workers/register",
        json={
            "registration": document.model_dump(mode="json"),
            "signature": base64.b64encode(sign_registration(private, document)).decode(),
            "public_key": base64.b64encode(private.public_key).decode(),
        },
    )


def test_controller_requires_enrollment_and_client_authentication() -> None:
    private = HybridSigningKey.generate()
    endpoint = "https://worker-a.invalid"
    app = create_controller_app(
        enrollments={"worker-a": WorkerEnrollment(private.public_key, endpoint)},
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
    private = HybridSigningKey.generate()
    app = create_controller_app(
        enrollments={"worker-a": WorkerEnrollment(private.public_key, "https://worker-a.invalid")},
        client_tokens={"secret": "client-a"},
    )
    with TestClient(app) as client:
        assert register(client, private, "https://attacker.invalid").status_code == 403
