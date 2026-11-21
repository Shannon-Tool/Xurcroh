from __future__ import annotations

import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.registry import WorkerRegistry
from xurcroh.controller.service import create_controller_app


def build_registry(private: Ed25519PrivateKey) -> WorkerRegistry:
    registration = Registration(
        worker_id="worker-a",
        endpoint="https://worker-a.invalid",
        models=["tiny"],
        modalities=["text"],
        timestamp=time.time(),
        engine="dense-stage",
    )
    registry = WorkerRegistry()
    registry.register(registration, sign_registration(private, registration), private.public_key())
    return registry


def test_heartbeat_refreshes_stale_worker_and_requires_authentication() -> None:
    private = Ed25519PrivateKey.generate()
    registry = build_registry(private)
    app = create_controller_app(registry, client_tokens={"secret": "client-a"})
    with TestClient(app, raise_server_exceptions=False) as client:
        record = registry.get("worker-a")
        record.last_heartbeat -= registry.registration_ttl_seconds + 5
        assert registry.healthy() == ()
        unauthenticated = client.post("/v1/workers/worker-a/heartbeat")
        assert unauthenticated.status_code == 403
        unknown = client.post(
            "/v1/workers/worker-b/heartbeat", headers={"Authorization": "Bearer secret"}
        )
        assert unknown.status_code >= 400
        response = client.post(
            "/v1/workers/worker-a/heartbeat", headers={"Authorization": "Bearer secret"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
        assert len(registry.healthy()) == 1
        assert registry.get("worker-a").last_heartbeat > time.time() - 5


def test_expired_circuits_are_purged_from_list_and_count() -> None:
    private = Ed25519PrivateKey.generate()
    registry = build_registry(private)
    app = create_controller_app(registry, circuit_ttl_seconds=0.05, require_authentication=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/circuits", json={"model": "tiny"})
        assert response.status_code == 200
        circuit_id = response.json()["circuit_id"]
        assert client.get("/metrics").json()["circuits_active"] == 1
        assert len(client.get("/v1/circuits").json()) == 1
        time.sleep(0.1)
        assert client.get("/metrics").json()["circuits_active"] == 0
        assert client.get("/v1/circuits").json() == []
        expired = client.post(f"/v1/relay/{circuit_id}")
        assert expired.status_code >= 400
        assert client.get("/metrics").json()["circuits_active"] == 0
