from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.native_stage import NativeStageServer
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler


def test_public_sdk_executes_dense_stage_over_native_quic(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xurcroh_quic

    master_key = bytes(range(32))
    compiled = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            master_key=master_key,
            seed=241,
        )
    ).compile()
    certificate, private_key = xurcroh_quic.generate_self_signed("localhost")
    certificate_path = tmp_path / "stage-ca.pem"
    certificate_path.write_bytes(certificate)
    server = NativeStageServer(
        compiled.manifest_path,
        stage_index=0,
        master_key=master_key,
        certificate=certificate,
        private_key=private_key,
    )
    server.start()
    monkeypatch.setenv("TEST_XURCROH_MASTER_KEY", master_key.hex())
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
                    "master_key_env": "TEST_XURCROH_MASTER_KEY",
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

    master_key = bytes(range(32))
    compiled = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            master_key=master_key,
            seed=241,
        )
    ).compile()
    monkeypatch.setenv("TEST_XURCROH_MASTER_KEY", master_key.hex())
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
            master_key_env="TEST_XURCROH_MASTER_KEY",
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
    try:
        with pytest.raises(RequestTimeoutError):
            engine._execute_stage(0, torch.randn(1, 16, dtype=torch.float64), "timeout", 0.01)
    finally:
        engine.close()
        server.close()
        thread.join(timeout=3)
