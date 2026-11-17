from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
import torch
import xurcroh_quic
from tests.fortress.conftest import MutualTLSIdentities
from tests.integration.test_ssh_executor import ssh_testbed


def test_datagram_relays_carry_real_mutual_tls_quic(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay, UdpDatagramRelay

    async def exercise() -> None:
        target = xurcroh_quic.TensorServer.bind_mutual(
            "127.0.0.1:0",
            [mtls_identities["server_certificate"], mtls_identities["ca"]],
            mtls_identities["server_key"],
            mtls_identities["ca"],
        )
        tcp = await TcpDatagramRelay.bind("127.0.0.1", 0, *target.local_addr().rsplit(":", 1))
        udp = await UdpDatagramRelay.bind("127.0.0.1", 0, "127.0.0.1", tcp.port)

        def echo() -> None:
            connection = target.accept()
            values = connection.receive()
            connection.send(*values[:3], values[3], values[4] or "", values[5], values[6])
            connection.close()

        thread = threading.Thread(target=echo, daemon=True)
        thread.start()

        def request() -> bytes:
            client = xurcroh_quic.TensorClient.connect_mutual(
                udp.address,
                "localhost",
                mtls_identities["ca"],
                [mtls_identities["client_certificate"], mtls_identities["ca"]],
                mtls_identities["client_key"],
            )
            client.send(bytes(16), 0, 0, "tensor", "u8", [4], b"test")
            response = client.receive()
            client.close()
            return response[-1]

        assert await asyncio.to_thread(request) == b"test"
        udp.close()
        tcp.close()
        target.close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    asyncio.run(exercise())


def test_failed_relay_start_cleans_up_the_remote_process(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.fortress import datagram_deploy

    commands: list[list[str]] = []
    monkeypatch.setattr(datagram_deploy, "start_detached", lambda worker, command: None)
    monkeypatch.setattr(
        datagram_deploy,
        "wait_remote_port",
        lambda worker, path: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    monkeypatch.setattr(
        datagram_deploy.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    worker = {
        "name": "worker",
        "host": "127.0.0.1",
        "ssh_user": "root",
        "ssh_port": 22,
    }

    with pytest.raises(TimeoutError):
        datagram_deploy.start_datagram_tunnels(
            [worker], str(tmp_path / "remote"), {"cluster": [{}]}, tmp_path
        )

    assert any("kill" in " ".join(command) for command in commands)


def test_tcp_relay_treats_clean_disconnect_as_normal() -> None:
    from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay

    class Writer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def exercise() -> None:
        relay = TcpDatagramRelay.__new__(TcpDatagramRelay)
        relay.target_host = "127.0.0.1"
        relay.target_port = 9
        reader = asyncio.StreamReader()
        reader.feed_eof()
        await relay.handle(reader, Writer())  # type: ignore[arg-type]

    asyncio.run(exercise())


def test_single_ssh_datagram_tunnel_executes_and_cleans_up(
    tmp_path: Path, mtls_identities: MutualTLSIdentities
) -> None:
    from xurcroh.fortress.datagram_deploy import (
        close_datagram_tunnels,
        start_datagram_tunnels,
    )

    with ssh_testbed(tmp_path) as base:
        assert base.ssh is not None
        target = xurcroh_quic.TensorServer.bind_mutual(
            "127.0.0.1:0",
            [mtls_identities["server_certificate"], mtls_identities["ca"]],
            mtls_identities["server_key"],
            mtls_identities["ca"],
        )

        def echo() -> None:
            connection = target.accept()
            values = connection.receive()
            connection.send(*values[:3], values[3], values[4] or "", values[5], values[6])
            connection.close()

        thread = threading.Thread(target=echo, daemon=True)
        thread.start()
        remote_root = tmp_path / "remote"
        remote_root.mkdir()
        (remote_root / "xurcroh").symlink_to(
            Path(__file__).parents[2] / "python" / "xurcroh", target_is_directory=True
        )
        worker = {
            "name": "relay-worker",
            "host": base.host,
            "ssh_user": base.ssh.username,
            "ssh_port": base.ssh.port,
            "ssh_key": base.ssh.key_path,
            "known_hosts": base.ssh.known_hosts_path,
            "python_path": sys.executable,
        }
        sdk = {"cluster": [{}]}
        try:
            start_datagram_tunnels(
                [worker],
                str(remote_root),
                sdk,
                tmp_path / "tunnels",
                target_ports=[int(target.local_addr().rsplit(":", 1)[1])],
                tunnel_name="single-relay",
            )
            endpoint = sdk["cluster"][0]["endpoint"].removeprefix("quic://")
            client = xurcroh_quic.TensorClient.connect_mutual(
                endpoint,
                "localhost",
                mtls_identities["ca"],
                [mtls_identities["client_certificate"], mtls_identities["ca"]],
                mtls_identities["client_key"],
            )
            client.send(bytes(16), 0, 0, "tensor", "u8", [4], b"test")
            response = client.receive()
            assert response[-1] == b"test"
            client.close()
        finally:
            close_datagram_tunnels([worker])
            target.close()
            thread.join(timeout=5)


def test_four_role_datagram_grid_returns_exact_product(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    from xurcroh.fortress.arithmetic import PRIME_MODULUS, split_secret
    from xurcroh.fortress.compiler import compile_operation
    from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay, UdpDatagramRelay
    from xurcroh.fortress.transport import (
        FortressRoleClient,
        MutualTLSIdentity,
        RoleEndpoint,
        encode_role_artifact,
    )
    from xurcroh.fortress.worker import FortressRoleWorker

    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    identity = MutualTLSIdentity(
        certificates=(mtls_identities["client_certificate"], mtls_identities["ca"]),
        private_key=mtls_identities["client_key"],
        pinned_ca=mtls_identities["ca"],
    )

    async def exercise() -> None:
        workers = []
        relays = []
        roles = []
        try:
            for activation in range(2):
                for weight in range(2):
                    recipient = X25519PrivateKey.generate()
                    public = recipient.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                    worker = FortressRoleWorker(
                        encode_role_artifact(operation, weight, public),
                        recipient.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                        certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                        private_key=mtls_identities["server_key"],
                        client_ca=mtls_identities["ca"],
                    )
                    worker.start()
                    workers.append(worker)
                    host, port = worker.address.rsplit(":", 1)
                    tcp = await TcpDatagramRelay.bind(host, 0, host, int(port))
                    udp = await UdpDatagramRelay.bind(host, 0, host, tcp.port)
                    relays.extend((tcp, udp))
                    roles.append(
                        RoleEndpoint(
                            activation, weight, udp.address, public, f"role-{activation}-{weight}"
                        )
                    )

            def request() -> None:
                client = FortressRoleClient(operation, roles, identity, cover_interval=60)
                shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
                expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(
                    PRIME_MODULUS
                )
                assert torch.equal(client.execute(shares), expected)
                client.close()

            await asyncio.to_thread(request)
        finally:
            for relay in reversed(relays):
                relay.close()
            for worker in workers:
                worker.close()

    asyncio.run(exercise())
