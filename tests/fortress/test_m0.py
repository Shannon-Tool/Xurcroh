from __future__ import annotations

import multiprocessing as mp
import time
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
from tests.fortress.evidence import exercise_adversarial_evidence, write_fortress_evidence
from tests.fortress.test_f0 import _role_process
from tests.models.test_dit_stage import tiny_dit_weights

from xurcroh.fortress.dit import FortressDiTCombiner, compile_dit_operations
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RoleEndpoint,
    encode_role_artifact,
)


def test_m0_synthetic_dit_runs_over_four_independent_processes(
    mtls_identities: MutualTLSIdentities,
) -> None:
    weights = tiny_dit_weights()
    operations = compile_dit_operations(weights)
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
                    for name, operation in operations.items()
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
                        f"m0-role-{activation_index}-{weight_index}",
                    )
                )
        identity = MutualTLSIdentity(
            certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
            private_key=mtls_identities["client_key"],
            pinned_ca=mtls_identities["ca"],
        )
        clients = {
            name: FortressRoleClient(operation, roles, identity, operation_name=name)
            for name, operation in operations.items()
        }
        combiner = FortressDiTCombiner(weights, operations, grid_clients=clients)
        latent = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=torch.float64)
        text = torch.tensor([[0.2, 0.1, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3]], dtype=torch.float64)
        expected = combiner.reference(latent, text, steps=2)
        inference_started = time.perf_counter()
        actual = combiner.run(latent, text, steps=2)
        inference_seconds = time.perf_counter() - inference_started
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        original = combiner.original_reference(latent, text, steps=2)
        assert torch.isfinite(actual - original).all()
        assert sum(client.request_count for client in clients.values()) == 328
        adversarial = exercise_adversarial_evidence(clients, roles, operations)
        assert sum(client.request_count for client in clients.values()) == 336
        assert sum(client.bytes_sent for client in clients.values()) > 0
        assert sum(client.bytes_received for client in clients.values()) > 0
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
    write_fortress_evidence(
        "M0",
        {
            "worker_processes": len(processes),
            "worker_exit_codes": [process.exitcode for process in processes],
            "operations": len(operations),
            "inference_role_requests": 328,
            "role_requests": sum(client.request_count for client in clients.values()),
            **adversarial,
            "bytes_sent": sum(client.bytes_sent for client in clients.values()),
            "bytes_received": sum(client.bytes_received for client in clients.values()),
            "inference_seconds": inference_seconds,
            "diffusion_steps": 2,
            "latent_tokens": latent.shape[0],
            "text_tokens": text.shape[0],
            "quantized_reference_max_abs_error": float((actual - expected).abs().max()),
            "original_model_max_abs_error": float((actual - original).abs().max()),
        },
    )
