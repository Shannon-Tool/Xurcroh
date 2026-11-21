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


def test_three_relay_chain_hides_worker_from_entry_and_client_from_exit(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from xurcroh.fortress.transport import RelayHop

    def raw_private(key: X25519PrivateKey) -> bytes:
        return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    def raw_public(key: X25519PrivateKey) -> bytes:
        return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )
    server = {
        "certificates": (mtls_identities["server_certificate"], mtls_identities["ca"]),
        "private_key": mtls_identities["server_key"],
        "client_ca": mtls_identities["ca"],
    }
    workers: list[FortressRoleWorker] = []
    worker_keys: list[tuple[X25519PrivateKey, X25519PrivateKey]] = []
    for _activation_index in range(2):
        for weight_index in range(2):
            recipient, onion = X25519PrivateKey.generate(), X25519PrivateKey.generate()
            worker = FortressRoleWorker(
                encode_role_artifact(operation, weight_index, raw_public(recipient)),
                raw_private(recipient),
                onion_private_key=raw_private(onion),
                allow_unattested=True,
                **server,
            )
            worker.start()
            workers.append(worker)
            worker_keys.append((recipient, onion))
    relay_keys = [X25519PrivateKey.generate() for _ in range(3)]
    exit_relay = ObliviousRelay(
        raw_private(relay_keys[2]),
        upstream_identity=identity,
        allowed_workers={f"quic://{worker.address}" for worker in workers},
        **server,
    )
    middle_relay = ObliviousRelay(
        raw_private(relay_keys[1]),
        upstream_identity=identity,
        allowed_workers={f"quic://{exit_relay.address}"},
        **server,
    )
    entry_relay = ObliviousRelay(
        raw_private(relay_keys[0]),
        upstream_identity=identity,
        allowed_workers={f"quic://{middle_relay.address}"},
        **server,
    )
    relays = [entry_relay, middle_relay, exit_relay]
    for relay in relays:
        relay.start()
    roles = []
    for index, worker in enumerate(workers):
        recipient, onion = worker_keys[index]
        route = RelayRoute(
            relay_address=entry_relay.address,
            worker_address=worker.address,
            relay_public_key=raw_public(relay_keys[0]),
            worker_public_key=raw_public(onion),
            extra_hops=(
                RelayHop(middle_relay.address, raw_public(relay_keys[1])),
                RelayHop(exit_relay.address, raw_public(relay_keys[2])),
            ),
        )
        assert route.hop_count == 3
        roles.append(
            RoleEndpoint(
                index // 2,
                index % 2,
                worker.address,
                raw_public(recipient),
                f"chain-role-{index}",
                route,
            )
        )
    client = FortressRoleClient(operation, roles, identity, cover_interval=60)
    shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    try:
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
        assert torch.equal(client.execute(shares), expected)
        assert [relay.forward_count for relay in relays] == [4, 4, 4]
        with pytest.raises(ProtocolError, match="authorized"):
            entry_relay._connect_upstream(f"quic://{workers[0].address}")
    finally:
        client.close()
        for relay in relays:
            relay.close()
        for worker in workers:
            worker.close()


def test_relay_route_is_built_from_worker_labels() -> None:
    from xurcroh.errors import ConfigurationError
    from xurcroh.fortress.transport import relay_route_from_labels

    hops = '[{"address":"10.0.0.1:9000","public_key":"' + "11" * 32 + '"},'
    hops += '{"address":"10.0.0.2:9000","public_key":"' + "22" * 32 + '"}]'
    labels = {"onion_public": "33" * 32, "relay_path": hops}

    route = relay_route_from_labels(labels, "10.0.0.9:9444", worker="w")

    assert route is not None
    assert route.onion_hops() == [
        (bytes.fromhex("11" * 32), "quic://10.0.0.2:9000"),
        (bytes.fromhex("22" * 32), "quic://10.0.0.9:9444"),
        (bytes.fromhex("33" * 32), ""),
    ]
    assert route.relay_address == "10.0.0.1:9000"
    assert relay_route_from_labels({}, "10.0.0.9:9444", worker="w") is None
    for broken in (
        {"onion_public": "33" * 32},
        {"relay_path": hops},
        {"onion_public": "33" * 32, "relay_path": "[]"},
        {"onion_public": "33" * 31, "relay_path": hops},
        {"onion_public": "33" * 32, "relay_path": '[{"address":"x"}]'},
        {"onion_public": "33" * 32, "relay_path": "not json"},
    ):
        with pytest.raises(ConfigurationError, match="relay"):
            relay_route_from_labels(broken, "10.0.0.9:9444", worker="w")


def test_public_sdk_generates_through_a_two_relay_onion_path(
    tiny_llama_path, tmp_path, mtls_identities: MutualTLSIdentities
) -> None:
    import json

    from Xurcroh import Xurcroh
    from xurcroh.fortress.model_compiler import compile_fortress_checkpoint

    def raw_private(key: X25519PrivateKey) -> bytes:
        return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    def raw_public(key: X25519PrivateKey) -> bytes:
        return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731, share_count=2)
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )
    server = {
        "certificates": (mtls_identities["server_certificate"], mtls_identities["ca"]),
        "private_key": mtls_identities["server_key"],
        "client_ca": mtls_identities["ca"],
    }
    for name in ("client_certificate", "client_key", "ca"):
        (tmp_path / name).write_bytes(mtls_identities[name])
    workers: list[FortressRoleWorker] = []
    cluster: list[dict[str, object]] = []
    for activation_index in range(2):
        for weight_index in range(2):
            recipient, onion = X25519PrivateKey.generate(), X25519PrivateKey.generate()
            worker = FortressRoleWorker(
                {
                    name: encode_role_artifact(operation, weight_index, raw_public(recipient))
                    for name, operation in compiled.operations.items()
                },
                raw_private(recipient),
                onion_private_key=raw_private(onion),
                allow_unattested=True,
                **server,
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
                        "recipient_public": raw_public(recipient).hex(),
                        "onion_public": raw_public(onion).hex(),
                        "cover_interval": "60",
                        "client_certificate_path": str(tmp_path / "client_certificate"),
                        "client_key_path": str(tmp_path / "client_key"),
                        "ca_path": str(tmp_path / "ca"),
                    },
                }
            )
    keys = [X25519PrivateKey.generate(), X25519PrivateKey.generate()]
    exit_relay = ObliviousRelay(
        raw_private(keys[1]),
        upstream_identity=identity,
        allowed_workers={f"quic://{worker.address}" for worker in workers},
        **server,
    )
    entry_relay = ObliviousRelay(
        raw_private(keys[0]),
        upstream_identity=identity,
        allowed_workers={f"quic://{exit_relay.address}"},
        **server,
    )
    path = json.dumps(
        [
            {"address": entry_relay.address, "public_key": raw_public(keys[0]).hex()},
            {"address": exit_relay.address, "public_key": raw_public(keys[1]).hex()},
        ]
    )
    for worker_config in cluster:
        worker_config["labels"]["relay_path"] = path  # type: ignore[index]
    entry_relay.start()
    exit_relay.start()
    model = {
        "path": str(tiny_llama_path),
        "revision": "a" * 40,
        "engine": "fortress",
        "fortress_seed": 731,
        "fortress_shares": 2,
    }
    try:
        with Xurcroh(cluster=cluster, model_path={"tiny": model}) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
            )
            trace = client.tasks.trace(response.execution.request_id)
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        counters = trace["transport"]["counters"]
        assert counters["fortress_role_requests"] > 0
        assert counters["fortress_verified_results"] == counters["fortress_role_requests"]
        assert trace["transport"]["frame_bytes"] == 16_384
        assert [route["relay_hops"] for route in trace["transport"]["routes"]] == [2, 2, 2, 2]
        assert {route["route"] for route in trace["transport"]["routes"]} == {"onion"}
        assert trace["model"]["share_count"] == 2
        assert trace["model"]["weight_bits"] == 4
        assert trace["model"]["weight_shares"].startswith("seed-expanded")
        assert trace["model"]["operations"] == len(compiled.operations)
        assert entry_relay.forward_count > 0
        assert entry_relay.forward_count == exit_relay.forward_count
    finally:
        entry_relay.close()
        exit_relay.close()
        for worker in workers:
            worker.close()
