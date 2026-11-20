from __future__ import annotations

from dataclasses import replace

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

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import PRIME_MODULUS, split_secret
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.relay import ObliviousRelay, RelayRoute
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RoleEndpoint,
    encode_role_artifact,
)
from xurcroh.fortress.worker import FortressRoleWorker


def test_oblivious_relay_forwards_four_exact_role_calls(
    mtls_identities: MutualTLSIdentities,
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )
    relay_private = X25519PrivateKey.generate()
    relay_public = relay_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    workers: list[FortressRoleWorker] = []
    routes: list[RelayRoute] = []
    roles: list[RoleEndpoint] = []
    for activation_index in range(2):
        for weight_index in range(2):
            recipient = X25519PrivateKey.generate()
            onion_private = X25519PrivateKey.generate()
            artifact = encode_role_artifact(
                operation,
                weight_index,
                recipient.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
            )
            worker = FortressRoleWorker(
                artifact,
                recipient.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                onion_private_key=onion_private.private_bytes(
                    Encoding.Raw, PrivateFormat.Raw, NoEncryption()
                ),
                allow_unattested=True,
            )
            worker.start()
            workers.append(worker)
            route = RelayRoute(
                relay_address="",
                worker_address=worker.address,
                relay_public_key=relay_public,
                worker_public_key=onion_private.public_key().public_bytes(
                    Encoding.Raw, PublicFormat.Raw
                ),
            )
            routes.append(route)
            roles.append(
                RoleEndpoint(
                    activation_index,
                    weight_index,
                    worker.address,
                    recipient.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
                    f"relay-role-{activation_index}-{weight_index}",
                    route,
                )
            )
    relay = ObliviousRelay(
        relay_private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
        certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["server_key"],
        client_ca=mtls_identities["ca"],
        upstream_identity=identity,
        allowed_workers={f"quic://{worker.address}" for worker in workers},
    )
    relay.start()
    with pytest.raises(ProtocolError, match="authorized"):
        relay._connect_upstream("quic://127.0.0.1:1")
    roles = [
        replace(role, relay=replace(role.relay, relay_address=relay.address)) for role in roles
    ]
    client = FortressRoleClient(operation, roles, identity, cover_interval=60)
    shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    try:
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
        assert torch.equal(client.execute(shares), expected)
        assert relay.forward_count == 4
    finally:
        client.close()
        relay.close()
        for worker in workers:
            worker.close()
