from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn

from shards.identities import StageIdentities
from Xurcroh import Xurcroh
from xurcroh.errors import RequestCancelledError
from xurcroh.kem import HybridKemKey


@contextmanager
def running_stage(
    manifest: Path,
    stage_index: int,
    private_key: HybridKemKey,
    attestation_document=None,
) -> Iterator[str]:
    from xurcroh.stage_worker import create_stage_app

    app = create_stage_app(
        manifest,
        stage_index=stage_index,
        private_key=private_key,
        attestation_document=attestation_document,
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        if time.monotonic() - started > 10:
            raise RuntimeError("stage worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http+dev://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_public_sdk_executes_real_encrypted_remote_stages(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = StageIdentities.create(2)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    with (
        running_stage(compiled.manifest_path, 0, identities.private_keys[0]) as first,
        running_stage(compiled.manifest_path, 1, identities.private_keys[1]) as second,
        Xurcroh(
            cluster=[
                {"name": "stage-0", "endpoint": first, "labels": {"stage_index": "0"}},
                {"name": "stage-1", "endpoint": second, "labels": {"stage_index": "1"}},
            ],
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
            pipeline_parallel=2,
            security={"mode": "development"},
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2,
            temperature=0,
        )
        assert response.execution.engine == "dense-stages"
        assert response.execution.worker_ids == ["stage-0", "stage-1"]
        assert response.usage.completion_tokens == 2
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2,
            temperature=0,
            stream=True,
        )
        chunks = list(stream)
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.completion_tokens == 2
        cancelling = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=4,
            temperature=0,
            stream=True,
            request_id="cancel-stage-request",
        )
        next(cancelling)
        assert client.cancel("cancel-stage-request")
        with pytest.raises(RequestCancelledError):
            list(cancelling)
        import httpx

        for endpoint in (first, second):
            capabilities = httpx.get(
                f"{endpoint.replace('http+dev://', 'http://')}/v1/capabilities"
            )
            assert capabilities.json()["cache_entries"] == 0
