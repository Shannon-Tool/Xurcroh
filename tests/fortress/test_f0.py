from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from multiprocessing.connection import Connection
from threading import Event

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.conftest import MutualTLSIdentities
from tests.fortress.test_combiner import build_combiner

from xurcroh.fortress.combiner import FortressCombiner
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RoleEndpoint,
    encode_role_artifact,
)


def _role_process(
    artifacts: Mapping[str, bytes],
    recipient_private_key: bytes,
    identities: MutualTLSIdentities,
    ready: Connection,
    stop: Event,
) -> None:
    from xurcroh.fortress.worker import FortressRoleWorker

    worker = FortressRoleWorker(
        artifacts,
        recipient_private_key,
        certificates=(identities["server_certificate"], identities["ca"]),
        private_key=identities["server_key"],
        client_ca=identities["ca"],
    )
    worker.start()
    ready.send(worker.address)
    stop.wait(60)
    worker.close()


def test_f0_full_model_runs_over_four_independent_processes(
    mtls_identities: MutualTLSIdentities,
) -> None:
    reference = build_combiner()
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    stops: list[Event] = []
    roles: list[RoleEndpoint] = []
    clients: dict[str, FortressRoleClient] = {}
    try:
        for activation_index in range(2):
            for weight_index in range(2):
                private = X25519PrivateKey.generate()
                public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                artifacts = {
                    name: encode_role_artifact(operation, weight_index, public)
                    for name, operation in reference.operations.items()
                }
                ready, child_ready = context.Pipe(duplex=False)
                stop = context.Event()
                process = context.Process(
                    target=_role_process,
                    args=(
                        artifacts,
                        private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                        mtls_identities,
                        child_ready,
                        stop,
                    ),
                )
                process.start()
                child_ready.close()
                assert ready.poll(15)
                address = ready.recv()
                ready.close()
                processes.append(process)
                stops.append(stop)
                roles.append(
                    RoleEndpoint(
                        activation_index,
                        weight_index,
                        address,
                        public,
                        f"f0-role-{activation_index}-{weight_index}",
                    )
                )
        identity = MutualTLSIdentity(
            certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
            private_key=mtls_identities["client_key"],
            pinned_ca=mtls_identities["ca"],
        )
        clients = {
            name: FortressRoleClient(operation, roles, identity, operation_name=name)
            for name, operation in reference.operations.items()
        }
        combiner = FortressCombiner(
            layers=reference.layers,
            operations=reference.operations,
            query_heads=reference.query_heads,
            kv_heads=reference.kv_heads,
            final_norm=reference.final_norm,
            lm_head=reference.lm_head,
            grid_clients=clients,
        )
        hidden = torch.tensor([[1, 0, -1, 2], [0, 2, 1, -1], [2, 1, 0, 1]], dtype=torch.float64)
        expected = reference.reference_forward(hidden)
        prefill = combiner.run_cached("f0-request", hidden[:2])
        decoded = combiner.run_cached("f0-request", hidden[2:])
        actual = torch.cat([prefill, decoded])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        assert (
            combiner.sample(actual[-1:]).item() == reference.reference_sample(expected[-1:]).item()
        )
        assert sum(client.request_count for client in clients.values()) == 168
        assert sum(client.bytes_sent for client in clients.values()) > 0
        assert sum(client.bytes_received for client in clients.values()) > 0
        combiner.drop_request("f0-request")
        assert combiner.cache_entries == 0
    finally:
        for client in clients.values():
            client.close()
        for stop in stops:
            stop.set()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        assert all(process.exitcode == 0 for process in processes)
