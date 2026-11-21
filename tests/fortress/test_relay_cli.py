from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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
from xurcroh.fortress.transport import (
    FortressRoleClient,
    MutualTLSIdentity,
    RelayRoute,
    RoleEndpoint,
    encode_role_artifact,
)
from xurcroh.fortress.worker import FortressRoleWorker


def raw_private(key: X25519PrivateKey) -> bytes:
    return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def raw_public(key: X25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def test_relay_cli_requires_an_allowlist(tmp_path: Path) -> None:
    from xurcroh.fortress.relay_cli import main

    with pytest.raises(SystemExit) as failure:
        main(["--onion-key", str(tmp_path / "k")])
    assert failure.value.code == 2


def test_relay_process_forwards_role_calls_and_stops_cleanly(
    tmp_path: Path, mtls_identities: MutualTLSIdentities
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    files = {
        "server.crt": mtls_identities["server_certificate"],
        "server.key": mtls_identities["server_key"],
        "client.crt": mtls_identities["client_certificate"],
        "client.key": mtls_identities["client_key"],
        "ca.crt": mtls_identities["ca"],
    }
    relay_key = X25519PrivateKey.generate()
    files["onion.key"] = raw_private(relay_key)
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    workers: list[FortressRoleWorker] = []
    keys: list[tuple[X25519PrivateKey, X25519PrivateKey]] = []
    for index in range(4):
        recipient, onion = X25519PrivateKey.generate(), X25519PrivateKey.generate()
        worker = FortressRoleWorker(
            encode_role_artifact(operation, index % 2, raw_public(recipient)),
            raw_private(recipient),
            certificates=(files["server.crt"], files["ca.crt"]),
            private_key=files["server.key"],
            client_ca=files["ca.crt"],
            onion_private_key=raw_private(onion),
            allow_unattested=True,
        )
        worker.start()
        workers.append(worker)
        keys.append((recipient, onion))
    ready = tmp_path / "ready"
    command = [sys.executable, "-m", "xurcroh.fortress.relay_cli"]
    command += ["--onion-key", str(tmp_path / "onion.key")]
    command += ["--certificate", str(tmp_path / "server.crt")]
    command += ["--private-key", str(tmp_path / "server.key"), "--ca", str(tmp_path / "ca.crt")]
    command += ["--upstream-certificate", str(tmp_path / "client.crt")]
    command += ["--upstream-private-key", str(tmp_path / "client.key")]
    command += ["--host", "127.0.0.1", "--port", "0", "--ready-file", str(ready)]
    for worker in workers:
        command += ["--allow", f"quic://{worker.address}"]
    process = subprocess.Popen(command, stderr=subprocess.PIPE)  # noqa: S603
    client = None
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert process.poll() is None, process.stderr.read().decode()  # type: ignore[union-attr]
            assert time.monotonic() < deadline, "relay did not become ready"
            time.sleep(0.05)
        address = ready.read_text().strip()
        identity = MutualTLSIdentity(
            certificates=(files["client.crt"], files["ca.crt"]),
            private_key=files["client.key"],
            pinned_ca=files["ca.crt"],
        )
        roles = [
            RoleEndpoint(
                index // 2,
                index % 2,
                worker.address,
                raw_public(keys[index][0]),
                f"cli-role-{index}",
                RelayRoute(
                    address, worker.address, raw_public(relay_key), raw_public(keys[index][1])
                ),
            )
            for index, worker in enumerate(workers)
        ]
        client = FortressRoleClient(operation, roles, identity, cover_interval=60)
        shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
        expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
        assert torch.equal(client.execute(shares), expected)
    finally:
        if client is not None:
            client.close()
        if process.poll() is None:
            os.kill(process.pid, signal.SIGTERM)
        code = process.wait(timeout=30)
        for worker in workers:
            worker.close()
    assert code == 0
