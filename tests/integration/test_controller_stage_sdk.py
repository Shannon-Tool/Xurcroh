from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from integration.test_stage_sdk import running_stage
from shards.identities import StageIdentities
from Xurcroh import Xurcroh
from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.registry import WorkerRegistry
from xurcroh.controller.service import create_controller_app


@contextmanager
def running_controller(worker_endpoint: str) -> Iterator[str]:
    private = Ed25519PrivateKey.generate()
    registration = Registration(
        worker_id="stage-0",
        endpoint=worker_endpoint,
        models=["tiny"],
        modalities=["text"],
        timestamp=time.time(),
        engine="dense-stage",
    )
    registry = WorkerRegistry()
    registry.register(registration, sign_registration(private, registration), private.public_key())
    app = create_controller_app(registry)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        if time.monotonic() - started > 10:
            raise RuntimeError("controller did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http+dev://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_public_sdk_uses_and_tears_down_controller_circuit(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    with (
        running_stage(compiled.manifest_path, 0, identities.private_keys[0]) as worker,
        running_controller(worker) as controller,
        Xurcroh(
            cluster={
                "workers": [{"name": "stage-0", "endpoint": "http+dev://127.0.0.1:1"}],
                "controller_endpoint": controller,
                "security": {"mode": "development"},
            },
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0,
        )
        assert response.usage.completion_tokens == 1
        from xurcroh.errors import RequestCancelledError

        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=4,
            temperature=0,
            stream=True,
            request_id="controller-cancel",
        )
        next(stream)
        assert client.cancel("controller-cancel")
        with pytest.raises(RequestCancelledError):
            list(stream)
        endpoint = controller.replace("http+dev://", "http://")
        assert httpx.get(f"{endpoint}/metrics").json()["circuits_active"] == 0
