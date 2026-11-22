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


def _signed_remote_grid(
    tiny_llama_path: Path,
    tmp_path: Path,
    mtls_identities: dict[str, bytes],
    shares: int,
    *,
    artifact_digest: str | None = None,
) -> tuple[list[dict], list]:
    """Start in-process role workers with signing identities; return the cluster and workers."""

    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from xurcroh.fortress.checkpoint import artifact_set_digest
    from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
    from xurcroh.fortress.transport import encode_role_artifact
    from xurcroh.fortress.worker import FortressRoleWorker
    from xurcroh.signatures import HybridSigningKey

    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731, share_count=shares)
    workers: list[FortressRoleWorker] = []
    cluster = []
    for activation_index in range(shares):
        for weight_index in range(shares):
            private = X25519PrivateKey.generate()
            identity = HybridSigningKey.generate()
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
                allow_unattested=True,
                identity_key=identity.to_bytes(),
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
                        "identity_public": identity.public_key.hex(),
                        "artifact_digest": artifact_digest or artifact_set_digest(artifacts),
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
    return cluster, workers


@pytest.mark.parametrize("shares", [2, 3])
def test_public_sdk_executes_remote_fortress_grid(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes], shares: int
) -> None:
    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, shares)
    try:
        with Xurcroh(
            cluster=cluster,
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "revision": "a" * 40,
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_shares": shares,
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
            assert grid is not None and len(grid.connections) == shares**2
            operation_clients = list(engine._clients.values())
            shared_clients = {
                id(transport.client)
                for operation in operation_clients
                for transport in operation._transports.values()
            }
            assert len(shared_clients) == shares**2
            import torch

            from xurcroh.fortress.arithmetic import FIELD_MODULUS

            operation = operation_clients[0]
            cycles = 65 if shares == 2 else 1
            for cycle in range(cycles):
                activation_shares = [
                    torch.randint(
                        0,
                        FIELD_MODULUS,
                        (operation.operation.quantized.dimensions[1],),
                        dtype=torch.int64,
                    )
                    for _ in range(shares)
                ]
                operation.execute(activation_shares, request_id=f"shared-rotation:{cycle}")
            assert all(connection.completed >= cycles for connection in grid.connections.values())
            from xurcroh.fortress.checkpoint import verify_checkpoint_signature
            from xurcroh.receipts import verify_trace

            trace = client.tasks.trace(response.execution.request_id)
            checkpoints = trace["transport"]["worker_checkpoints"]
            assert sorted(item["worker"] for item in checkpoints) == sorted(
                response.execution.worker_ids
            )
            for item in checkpoints:
                body = verify_checkpoint_signature(item)
                assert body["calls"] > 0 and item["artifacts_pinned"] is True
            assert verify_trace(trace)["worker_checkpoints"] == shares**2
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        assert len(response.execution.worker_ids) == shares**2
        metrics = client.metrics()
        assert metrics["fortress_bytes_sent"] > 0
        assert metrics["fortress_bytes_received"] > 0
        assert metrics["fortress_role_requests"] > 0
        assert metrics["fortress_reconnections"] >= 0
        assert metrics["fortress_cover_failures"] >= 0
    finally:
        for worker in workers:
            worker.close()


def test_public_sdk_executes_split_operation_grids(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from xurcroh.config import WorkerConfig
    from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
    from xurcroh.fortress.placement import (
        Operation,
        PlacementPolicy,
        Provider,
        apply_placement,
        place_operations,
    )
    from xurcroh.fortress.transport import encode_role_artifact
    from xurcroh.fortress.worker import FortressRoleWorker

    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    operation_names = list(compiled.operations)
    worker_names = [f"grid-{group}-{index}" for group in range(2) for index in range(4)]
    placement = place_operations(
        [Operation(name, index, "projection") for index, name in enumerate(operation_names)],
        [
            Provider(
                name,
                "us" if name.startswith("grid-0") else "de",
                "amd" if name.startswith("grid-0") else "intel",
            )
            for name in worker_names
        ],
        PlacementPolicy(shares=2, max_provider_fraction=1.0),
    )
    workers: list[FortressRoleWorker] = []
    cluster = []
    for group in range(2):
        for index in range(4):
            name = f"grid-{group}-{index}"
            activation_index, weight_index = divmod(index, 2)
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            assignments = {
                assignment.unit_id: assignment
                for assignment in placement.assignments
                if assignment.provider_id == name
            }
            artifacts = {
                operation_name: encode_role_artifact(
                    operation, assignments[operation_name].weight_share, public
                )
                for operation_name, operation in compiled.operations.items()
                if operation_name in assignments
            }
            worker = FortressRoleWorker(
                artifacts,
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                allow_unattested=True,
            )
            worker.start()
            workers.append(worker)
            cluster.append(
                WorkerConfig(
                    name=name,
                    endpoint=f"quic://{worker.address}",
                    gpu={"model": "test-gpu", "memory_gb": 1},
                    labels={
                        "activation_index": str(activation_index),
                        "weight_index": str(weight_index),
                        "recipient_public": public.hex(),
                        "cover_interval": "60",
                    },
                )
            )
    (tmp_path / "client.crt").write_bytes(mtls_identities["client_certificate"])
    (tmp_path / "client.key").write_bytes(mtls_identities["client_key"])
    (tmp_path / "ca.crt").write_bytes(mtls_identities["ca"])
    cluster = apply_placement(cluster, placement)
    for worker in cluster:
        worker.labels.update(
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
            assert len(engine._grid_runtimes) >= 2
            assert all(len(runtime.connections) == 4 for runtime in engine._grid_runtimes)
        assert response.usage.completion_tokens == 1
        assert len(response.execution.worker_ids) == 8
    finally:
        for worker in workers:
            worker.close()


def test_fortress_engine_selects_operation_specific_roles(tiny_llama_path: Path) -> None:
    from xurcroh.config import ModelDefinition, WorkerConfig
    from xurcroh.fortress.engine import FortressEngine

    workers = [
        WorkerConfig(
            name=name,
            endpoint=f"quic://{name}:9444",
            labels={
                "activation_index": "0",
                "weight_index": "0",
                "recipient_public": "00" * 32,
                "operations": operation,
            },
        )
        for name, operation in (("grid-a", "operation-a"), ("grid-b", "operation-b"))
    ]
    engine = FortressEngine(
        "tiny",
        ModelDefinition(path=str(tiny_llama_path), engine="fortress"),
        workers,
    )

    assert [role.provider_id for role in engine._roles("operation-a")] == ["grid-a"]
    assert [role.provider_id for role in engine._roles("operation-b")] == ["grid-b"]


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


def test_request_fails_when_a_worker_holds_other_artifacts_than_deployed(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    from xurcroh.errors import ProtocolError

    cluster, workers = _signed_remote_grid(
        tiny_llama_path, tmp_path, mtls_identities, 2, artifact_digest="0" * 64
    )
    try:
        with (
            Xurcroh(
                cluster=cluster,
                model_path={
                    "tiny": {
                        "path": str(tiny_llama_path),
                        "revision": "a" * 40,
                        "engine": "fortress",
                        "fortress_seed": 731,
                    }
                },
            ) as client,
            pytest.raises(ProtocolError, match="artifacts other than the ones deployed"),
        ):
            client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
            )
    finally:
        for worker in workers:
            worker.close()
