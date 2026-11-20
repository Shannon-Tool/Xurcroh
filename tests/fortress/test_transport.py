from __future__ import annotations

import json
import time

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.fortress.arithmetic import PRIME_MODULUS, split_secret
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.grid import RoleExecutionError
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RoleEndpoint,
    combine_role_results,
    decode_role_artifact,
    encode_role_artifact,
)
from xurcroh.fortress.worker import FortressRoleWorker


def test_role_artifact_reconstructs_only_its_own_weight_share() -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    encoded = encode_role_artifact(operation, weight_index=1, recipient_public=bytes(range(32)))
    artifact = decode_role_artifact(encoded)
    assert artifact.weight_index == 1
    assert torch.equal(artifact.matrix, operation.shares[1].matrix)
    assert artifact.recipient_public == bytes(range(32))
    document = json.loads(encoded)
    assert document["format"] == "xurcroh-fortress-role-v1"
    assert "challenge" not in document


def test_seeded_role_artifact_expands_without_storing_the_weight_matrix() -> None:
    operation = compile_operation(
        torch.arange(48, dtype=torch.int64).reshape(6, 8),
        threshold=2,
        slice_count=3,
        share_count=3,
        seed=bytes(range(32)),
        challenge_seed=19,
    )
    encoded = encode_role_artifact(operation, weight_index=0, recipient_public=bytes(range(32)))
    document = json.loads(encoded)
    assert document["format"] == "xurcroh-fortress-role-v2"
    assert "storage" not in document
    artifact = decode_role_artifact(encoded)
    assert torch.equal(artifact.matrix, operation.shares[0].matrix)
    final = json.loads(
        encode_role_artifact(operation, weight_index=2, recipient_public=bytes(range(32)))
    )
    assert final["format"] == "xurcroh-fortress-role-v1"
    assert "storage" in final


def test_grid_runtime_rejects_duplicate_role_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xurcroh.fortress import transport

    monkeypatch.setattr(transport, "_SharedRoleConnection", lambda *args: object())
    identity = MutualTLSIdentity(certificates=(b"ca",), private_key=b"key", pinned_ca=b"ca")
    roles = [
        RoleEndpoint(0, 0, "127.0.0.1:9444", bytes(32), "duplicate"),
        RoleEndpoint(0, 0, "127.0.0.1:9445", bytes(32), "duplicate"),
    ]
    with pytest.raises(ValueError, match="duplicate role coordinates"):
        transport.FortressGridRuntime(roles, identity)


def test_direct_grid_recovers_after_role_connection_loss(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from xurcroh.fortress.transport import FortressGridRuntime

    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
        server_name="localhost",
    )
    records = []
    roles = []
    recovered = []

    def start(record) -> None:
        if "recipient_key" not in record:
            private = X25519PrivateKey.generate()
            record["recipient_key"] = private.private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()
            )
            record["public"] = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        artifact = encode_role_artifact(operation, record["weight_index"], record["public"])
        worker = FortressRoleWorker(
            {"recovery": artifact},
            record["recipient_key"],
            certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
            private_key=mtls_identities["server_key"],
            client_ca=mtls_identities["ca"],
            address=record.get("address", "127.0.0.1:0"),
        )
        worker.start()
        record.update(worker=worker, address=worker.address)

    def recover(provider_id: str, generation: int) -> int:
        recovered.append(provider_id)
        start(records[0])
        return generation + 1

    for activation_index in range(2):
        for weight_index in range(2):
            record = {"weight_index": weight_index}
            start(record)
            records.append(record)
            roles.append(
                RoleEndpoint(
                    activation_index,
                    weight_index,
                    record["address"],
                    record["public"],
                    f"role-{weight_index}",
                )
            )
    runtime = FortressGridRuntime(roles, identity, cover_interval=60, recover_role=recover)
    try:
        client = runtime.client(operation, "recovery")
        shares = split_secret(torch.tensor([5, 7]), count=2).shares
        expected = client.execute(shares, request_id="before")
        records[0]["worker"].close()
        actual = client.execute(shares, request_id="after")
        assert torch.equal(actual, expected)
        assert recovered == ["role-0"]
    finally:
        runtime.close()
        for record in records:
            record["worker"].close()


def test_real_grid_transport_is_exact_and_isolates_duplicate_request_ids(
    mtls_identities: MutualTLSIdentities,
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
        server_name="localhost",
    )
    workers: list[FortressRoleWorker] = []
    roles: list[RoleEndpoint] = []
    for activation_index in range(2):
        for weight_index in range(2):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            artifact = encode_role_artifact(operation, weight_index, public)
            worker = FortressRoleWorker(
                artifact,
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                rotation_interval=1,
            )
            worker.start()
            workers.append(worker)
            roles.append(
                RoleEndpoint(
                    activation_index=activation_index,
                    weight_index=weight_index,
                    address=worker.address,
                    recipient_public=public,
                    provider_id=f"role-{activation_index}-{weight_index}",
                )
            )
    clients = [
        FortressRoleClient(operation, roles, identity, rotation_interval=1),
        FortressRoleClient(operation, roles, identity, rotation_interval=1, cover_interval=0.01),
    ]
    shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    try:
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
        for client in clients:
            assert torch.equal(client.execute(shares, request_id="duplicate"), expected)
        clients[0].close()
        assert torch.equal(clients[1].execute(shares, request_id="duplicate"), expected)
        assert clients[1].request_count == 8
        assert clients[1].bytes_sent > 0
        assert clients[1].bytes_received > 0
        assert clients[0].frame_sizes == {16_384}
        time.sleep(0.05)
        assert clients[1].cover_request_count > 0
        assert clients[1].cover_bytes_sent > 0
        from xurcroh.fortress.attacks import measure_timing_classifier

        classifier = measure_timing_classifier(clients[1].timing_observations)
        assert classifier.cover_samples > 0
        assert classifier.real_samples == 8
        assert 0.0 <= classifier.leave_one_out_accuracy <= 1.0
        assert 0.0 <= classifier.leave_one_out_balanced_accuracy <= 1.0
        from dataclasses import asdict

        from tests.fortress.evidence import write_fortress_evidence

        write_fortress_evidence(
            "TRANSPORT",
            {
                "execution": "local CPU; four in-process role workers; real Rust QUIC",
                "fixed_frame_bytes": sorted(clients[0].frame_sizes),
                "cover_request_count": clients[1].cover_request_count,
                "cover_bytes_sent": clients[1].cover_bytes_sent,
                "cover_bytes_received": clients[1].cover_bytes_received,
                "timing_classifier": asdict(classifier),
            },
        )
    finally:
        for client in clients:
            client.close()
        for worker in workers:
            worker.close()
    assert all(worker.active_sessions == 0 for worker in workers)
    assert all(worker.active_requests == 0 for worker in workers)


def test_three_by_three_grid_executes_exactly(
    mtls_identities: MutualTLSIdentities,
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        threshold=2,
        slice_count=3,
        share_count=3,
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )
    workers: list[FortressRoleWorker] = []
    roles: list[RoleEndpoint] = []
    client = None
    try:
        for activation_index in range(3):
            for weight_index in range(3):
                private = X25519PrivateKey.generate()
                public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                worker = FortressRoleWorker(
                    encode_role_artifact(operation, weight_index, public),
                    private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                    certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                    private_key=mtls_identities["server_key"],
                    client_ca=mtls_identities["ca"],
                )
                worker.start()
                workers.append(worker)
                roles.append(
                    RoleEndpoint(
                        activation_index,
                        weight_index,
                        worker.address,
                        public,
                        f"n3-role-{activation_index}-{weight_index}",
                    )
                )
        client = FortressRoleClient(operation, roles, identity, cover_interval=60)
        shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=3).shares
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
        assert torch.equal(client.execute(shares), expected)
        assert client.request_count == 9
    finally:
        if client is not None:
            client.close()
        for worker in workers:
            worker.close()


def test_client_verifies_exact_products_and_attributes_a_malicious_role() -> None:
    import pytest

    from xurcroh.fortress.grid import RoleExecutionError
    from xurcroh.fortress.transport import combine_role_results

    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    activation_shares = split_secret(torch.tensor([5, 7]), count=2).shares
    results = {
        (activation_index, weight_index): [
            sum(
                int(activation) * int(value)
                for activation, value in zip(
                    activation_shares[activation_index].remainder(PRIME_MODULUS), row, strict=True
                )
            )
            for row in share.matrix
        ]
        for (activation_index, weight_index), share in (
            (key, operation.shares[key[1]])
            for key in (
                (activation_index, weight_index)
                for activation_index in range(2)
                for weight_index in range(2)
            )
        )
    }
    providers = {key: f"role-{key[0]}-{key[1]}" for key in results}
    results[(0, 0)][0] += 1
    with pytest.raises(RoleExecutionError, match="role-0-0"):
        combine_role_results(activation_shares, operation.shares, results, providers)


def test_prime_multiple_forgery_cannot_change_reconstructed_field_result() -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    activation_shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    results = {
        (activation_index, weight_index): [
            sum(
                int(activation) * int(value)
                for activation, value in zip(
                    activation_shares[activation_index].remainder(PRIME_MODULUS), row, strict=True
                )
            )
            % PRIME_MODULUS
            for row in share.matrix
        ]
        for (activation_index, weight_index), share in (
            (key, operation.shares[key[1]])
            for key in (
                (activation_index, weight_index)
                for activation_index in range(2)
                for weight_index in range(2)
            )
        )
    }
    providers = {key: f"role-{key[0]}-{key[1]}" for key in results}
    results[(0, 0)][0] += PRIME_MODULUS
    with pytest.raises(RoleExecutionError, match="role-0-0"):
        combine_role_results(activation_shares, operation.shares, results, providers)


def test_integrity_failure_quarantines_the_role_client(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from xurcroh.errors import ProtocolError

    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )
    workers: list[FortressRoleWorker] = []
    roles: list[RoleEndpoint] = []
    client = None
    try:
        for activation_index in range(2):
            for weight_index in range(2):
                private = X25519PrivateKey.generate()
                public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                worker = FortressRoleWorker(
                    encode_role_artifact(operation, weight_index, public),
                    private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                    certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                    private_key=mtls_identities["server_key"],
                    client_ca=mtls_identities["ca"],
                )
                worker.start()
                workers.append(worker)
                roles.append(
                    RoleEndpoint(
                        activation_index,
                        weight_index,
                        worker.address,
                        public,
                        f"quarantine-{activation_index}-{weight_index}",
                    )
                )
        client = FortressRoleClient(operation, roles, identity, cover_interval=60)
        operation.shares[0].challenge.v.add_(1)
        shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
        with pytest.raises(RoleExecutionError):
            client.execute(shares)
        with pytest.raises(ProtocolError, match="quarantined"):
            client.execute(shares)
    finally:
        if client is not None:
            client.close()
        for worker in workers:
            worker.close()


def test_field_activation_wire_format_uses_four_bytes() -> None:
    from types import SimpleNamespace

    worker = FortressRoleWorker.__new__(FortressRoleWorker)
    worker.artifacts = {"op": SimpleNamespace(matrix=torch.zeros(2, 2, dtype=torch.int64))}
    payload = b"\x02\x00op" + (5).to_bytes(4, "little") + (7).to_bytes(4, "little")
    name, activation = worker._decode_activation(payload)
    assert name == "op"
    assert activation.tolist() == [5, 7]


def test_cuda_worker_canonicalizes_signed_field_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from xurcroh.fortress.arithmetic import FIELD_MODULUS

    class Controller:
        def matvec_device(
            self, weights: torch.Tensor, activation: torch.Tensor, output: torch.Tensor
        ) -> None:
            output.fill_(-6)

    worker = FortressRoleWorker.__new__(FortressRoleWorker)
    worker.artifacts = {"op": SimpleNamespace(matrix=torch.zeros(1, 1, dtype=torch.int64))}
    worker._cuda_controller = Controller()
    worker._cuda_weights = {"op": torch.ones(1, 1, dtype=torch.int32)}
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    original_empty = torch.empty

    def empty_without_device(*args: int, **kwargs: object) -> torch.Tensor:
        options = dict(kwargs)
        options.pop("device", None)
        return original_empty(*args, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "empty", empty_without_device)
    assert worker._execute_activation("op", torch.tensor([1])) == [FIELD_MODULUS - 1]


def test_role_transport_reconnects_once_after_timeout() -> None:
    import threading

    from xurcroh.fortress.transport import _RoleTransport

    transport = _RoleTransport.__new__(_RoleTransport)
    transport._call_lock = threading.Lock()
    transport._execution_lock = transport._call_lock
    transport._connection = None
    transport.role = RoleEndpoint(0, 0, "localhost:9444", bytes(range(32)), "test-role")
    transport.reconnect_count = 0
    transport.observations = []
    calls = []
    reconnects = []

    def execute(activation, request_id, sequence, activation_seed=None):
        del activation_seed
        calls.append(sequence)
        if len(calls) == 1:
            raise ValueError("QUIC connection failed: timed out")
        return [7], 10, 20

    transport._execute = execute
    transport._connect = lambda: reconnects.append(True)
    assert transport.execute(torch.tensor([1]), "request", 3) == ([7], 10, 20)
    assert calls == [3, 3]
    assert reconnects == [True]
    assert transport.reconnect_count == 1


def test_seeded_activation_wire_format_expands_exact_share() -> None:
    from types import SimpleNamespace

    from xurcroh.fortress.arithmetic import _seeded_uniform

    worker = FortressRoleWorker.__new__(FortressRoleWorker)
    worker.artifacts = {"op": SimpleNamespace(matrix=torch.zeros(2, 3, dtype=torch.int64))}
    worker._cuda_controller = None
    seed = bytes(range(32))
    payload = b"XF2\x01\x02\x00op" + seed
    name, activation = worker._decode_activation(payload)
    assert name == "op"
    assert torch.equal(activation, _seeded_uniform(seed, (3,)))


def test_execution_health_expires_and_cancels() -> None:
    import threading

    from xurcroh.errors import RequestCancelledError, RequestTimeoutError
    from xurcroh.fortress.transport import check_execution_health, execution_health

    with (
        execution_health(threading.Event(), time.monotonic() - 0.001),
        pytest.raises(RequestTimeoutError, match="deadline"),
    ):
        check_execution_health()

    cancelled = threading.Event()
    cancelled.set()
    with (
        execution_health(cancelled, time.monotonic() + 60),
        pytest.raises(RequestCancelledError, match="cancelled"),
    ):
        check_execution_health()


def test_role_client_checks_execution_health_between_rows() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from xurcroh.errors import RequestTimeoutError
    from xurcroh.fortress.transport import execution_health

    client = FortressRoleClient.__new__(FortressRoleClient)
    client.operation = SimpleNamespace(shares={0: None, 1: None})
    client.seeded_activations = False
    client._runtime = None
    client._closed = False
    client._quarantined = None
    client._cover_error = None
    client._lock = threading.Lock()
    client.request_count = 0
    client.bytes_sent = 0
    client.bytes_received = 0
    client.cover_request_count = 0
    client.cover_bytes_sent = 0
    client.cover_bytes_received = 0
    client._executor = ThreadPoolExecutor(max_workers=1)
    client._transports = {}

    def execute_token(
        activation_shares, request_id, sequence, *, cover=False, activation_seed=None
    ):
        del activation_shares, request_id, cover, activation_seed
        time.sleep(0.08)
        return torch.tensor([[sequence]], dtype=torch.int64)

    client._execute_token = execute_token
    shares = [
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.tensor([[5, 6], [7, 8]], dtype=torch.int64),
    ]
    try:
        with (
            execution_health(threading.Event(), time.monotonic() + 0.05),
            pytest.raises(RequestTimeoutError, match="deadline"),
        ):
            client.execute(shares, request_id="row-timeout")
    finally:
        client._executor.shutdown(wait=True)
