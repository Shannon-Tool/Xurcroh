from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import Xurcroh


@pytest.mark.parametrize("shares", [2, 3])
def test_public_sdk_executes_fortress_checkpoint(tiny_llama_path: Path, shares: int) -> None:
    with Xurcroh(
        model_path={
            "tiny": {
                "path": str(tiny_llama_path),
                "engine": "fortress",
                "fortress_shares": shares,
            },
        }
    ) as client:
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=1,
        )
    assert response.choices[0].message.content is not None
    assert response.usage.completion_tokens == 1
    assert response.execution.engine == "fortress-local"
    assert response.execution.worker_ids == ["local"]


def test_public_sdk_executes_remote_fortress_grid(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
    from xurcroh.fortress.transport import encode_role_artifact
    from xurcroh.fortress.worker import FortressRoleWorker

    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    workers: list[FortressRoleWorker] = []
    cluster = []
    for activation_index in range(2):
        for weight_index in range(2):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            artifacts = {
                name: encode_role_artifact(operation, weight_index, public)
                for name, operation in compiled.operations.items()
            }
            worker = FortressRoleWorker(
                artifacts,
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
            )
            worker.start()
            workers.append(worker)
            cluster.append(
                {
                    "name": f"fortress-{activation_index}-{weight_index}",
                    "endpoint": f"quic://{worker.address}",
                    "gpu": {"model": "test-gpu", "memory_gb": 1},
                    "labels": {
                        "activation_index": str(activation_index),
                        "weight_index": str(weight_index),
                        "recipient_public": public.hex(),
                        "cover_interval": "60",
                    },
                }
            )
    (tmp_path / "client.crt").write_bytes(mtls_identities["client_certificate"])
    (tmp_path / "client.key").write_bytes(mtls_identities["client_key"])
    (tmp_path / "ca.crt").write_bytes(mtls_identities["ca"])
    for worker in cluster:
        worker["labels"].update(
            {
                "client_certificate_path": str(tmp_path / "client.crt"),
                "client_key_path": str(tmp_path / "client.key"),
                "ca_path": str(tmp_path / "ca.crt"),
            }
        )
    try:
        with Xurcroh(
            cluster=cluster,
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "revision": "a" * 40,
                    "engine": "fortress",
                    "fortress_seed": 731,
                }
            },
        ) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
            )
            engine = client._runtime._engines["tiny"]
            grid = engine._grid_runtime
            assert grid is not None and len(grid.connections) == 4
            operation_clients = list(engine._clients.values())
            shared_clients = {
                id(transport.client)
                for operation in operation_clients
                for transport in operation._transports.values()
            }
            assert len(shared_clients) == 4
            import torch

            from xurcroh.fortress.arithmetic import FIELD_MODULUS

            operation = operation_clients[0]
            for cycle in range(65):
                shares = [
                    torch.randint(
                        0,
                        FIELD_MODULUS,
                        (operation.operation.quantized.dimensions[1],),
                        dtype=torch.int64,
                    )
                    for _ in range(2)
                ]
                operation.execute(shares, request_id=f"shared-rotation:{cycle}")
            assert all(connection.completed >= 65 for connection in grid.connections.values())
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        assert len(response.execution.worker_ids) == 4
        metrics = client.metrics()
        assert metrics["fortress_bytes_sent"] > 0
        assert metrics["fortress_bytes_received"] > 0
        assert metrics["fortress_role_requests"] > 0
        assert metrics["fortress_reconnections"] >= 0
        assert metrics["fortress_cover_failures"] >= 0
    finally:
        for worker in workers:
            worker.close()


def test_fortress_inference_queue_honors_request_timeout(tiny_llama_path: Path) -> None:

    import pytest

    from xurcroh.errors import RequestTimeoutError

    with Xurcroh(
        model_path={"tiny": {"path": str(tiny_llama_path), "engine": "fortress"}}
    ) as client:
        engine = client._runtime._engine("tiny")
        engine._inference_lock.acquire()
        try:
            with pytest.raises(RequestTimeoutError, match="inference queue"):
                client.chat.completions.create(
                    model="tiny",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=1,
                    timeout=0.01,
                )
        finally:
            engine._inference_lock.release()
