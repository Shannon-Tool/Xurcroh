from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shards.identities import StageIdentities
from Xurcroh import Xurcroh
from xurcroh.native_stage import NativeStageServer


def test_public_sdk_executes_dense_stage_over_native_quic(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xurcroh_quic

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    certificate, private_key = xurcroh_quic.generate_self_signed("localhost")
    certificate_path = tmp_path / "stage-ca.pem"
    certificate_path.write_bytes(certificate)
    server = NativeStageServer(
        compiled.manifest_path,
        stage_index=0,
        stage_private_key=identities.private_keys[0],
        certificate=certificate,
        private_key=private_key,
    )
    server.start()
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    try:
        with Xurcroh(
            cluster={
                "name": "localhost",
                "endpoint": f"quic://{server.address}",
                "labels": {"ca_path": str(certificate_path), "server_name": "localhost"},
            },
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
            security={"mode": "development"},
        ) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
                temperature=0,
            )
        assert response.usage.completion_tokens == 1
        assert response.execution.worker_ids == ["localhost"]
        assert server.stage.sessions.active_count == 0
    finally:
        server.close()


def test_native_stage_timeout_aborts_wait(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    import torch
    import xurcroh_quic

    from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
    from xurcroh.errors import RequestTimeoutError
    from xurcroh.runtime.stage import DenseStageEngine
    from xurcroh.runtime.stage_sessions import ClientStageSession

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    certificate, private_key = xurcroh_quic.generate_self_signed("localhost")
    server = xurcroh_quic.TensorServer("127.0.0.1:0", certificate, private_key)
    address = server.local_addr()

    def no_response() -> None:
        connection = server.accept()
        connection.receive()
        time.sleep(2)

    thread = threading.Thread(target=no_response, daemon=True)
    thread.start()
    engine = DenseStageEngine(
        "tiny",
        ModelDefinition(
            path=str(tiny_llama_path),
            manifest_path=str(compiled.manifest_path),
            boundary_key_env="TEST_XURCROH_CLIENT_BOUNDARY_KEY",
        ),
        [
            WorkerConfig(
                name="localhost",
                endpoint=f"quic://{address}",
                labels={"ca_path": str(tmp_path / "ca.pem"), "server_name": "localhost"},
            )
        ],
        SecurityConfig(mode="development"),
    )
    (tmp_path / "ca.pem").write_bytes(certificate)
    session = ClientStageSession(0, identities.recipients[0].public_key)
    try:
        with pytest.raises(RequestTimeoutError, match="native stage 0 timed out"):
            engine._execute_stage(
                0,
                torch.randn(1, 16, dtype=torch.float64),
                "timeout",
                session,
                0.01,
            )
    finally:
        engine.close()
        server.close()
        thread.join(timeout=3)


def test_stage_zeroization_accepts_inference_tensors() -> None:
    from xurcroh.stage_worker import _zero_tensors

    with torch.inference_mode():
        first = torch.ones(2, 2)
        second = torch.ones(3)
    _zero_tensors(first, second)
    assert not first.any()
    assert not second.any()
