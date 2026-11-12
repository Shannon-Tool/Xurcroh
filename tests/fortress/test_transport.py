from __future__ import annotations

import json

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.fortress.arithmetic import RING_MODULUS, split_secret
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RoleEndpoint,
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
        FortressRoleClient(operation, roles, identity),
        FortressRoleClient(operation, roles, identity),
    ]
    shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    try:
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(RING_MODULUS)
        for client in clients:
            assert torch.equal(client.execute(shares, request_id="duplicate"), expected)
        clients[0].close()
        assert torch.equal(clients[1].execute(shares, request_id="duplicate"), expected)
        assert clients[1].request_count == 8
        assert clients[1].bytes_sent > 0
        assert clients[1].bytes_received > 0
    finally:
        for client in clients:
            client.close()
        for worker in workers:
            worker.close()
    assert all(worker.active_sessions == 0 for worker in workers)
    assert all(worker.active_requests == 0 for worker in workers)


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
                    activation_shares[activation_index].remainder(RING_MODULUS), row, strict=True
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
