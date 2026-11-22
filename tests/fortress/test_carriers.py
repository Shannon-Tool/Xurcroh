from __future__ import annotations

import asyncio
import threading

import pytest
import xurcroh_quic
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.errors import ConfigurationError, ProtocolError
from xurcroh.fortress.carriers import (
    CarrierRelay,
    CarrierSpec,
    SamDialer,
    Socks5Dialer,
    carrier_from_labels,
)
from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay


class FakeTor:
    """A SOCKS5 proxy that records what it was asked and connects like Tor would."""

    def __init__(self, onion_targets: dict[str, tuple[str, int]], *, status: int = 0) -> None:
        self.onion_targets = onion_targets
        self.status = status
        self.requests: list[tuple[str, int, bytes, bytes]] = []
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return int(self.server.sockets[0].getsockname()[1])

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        assert version == 5
        user = password = b""
        if 2 in methods:
            writer.write(b"\x05\x02")
            assert (await reader.readexactly(1)) == b"\x01"
            user = await reader.readexactly((await reader.readexactly(1))[0])
            password = await reader.readexactly((await reader.readexactly(1))[0])
            writer.write(b"\x01\x00")
        else:
            writer.write(b"\x05\x00")
        header = await reader.readexactly(4)
        assert header[:3] == b"\x05\x01\x00"
        # A client that resolved the name itself would send address type 1, not 3.
        assert header[3] == 3, "hostname must reach the proxy unresolved"
        name = (await reader.readexactly((await reader.readexactly(1))[0])).decode()
        port = int.from_bytes(await reader.readexactly(2), "big")
        self.requests.append((name, port, user, password))
        if self.status or name not in self.onion_targets:
            writer.write(bytes([5, self.status or 4, 0, 1]) + bytes(6))
            await writer.drain()
            writer.close()
            return
        upstream_reader, upstream_writer = await asyncio.open_connection(*self.onion_targets[name])
        writer.write(b"\x05\x00\x00\x01" + bytes(6))
        await writer.drain()
        await _splice(reader, writer, upstream_reader, upstream_writer)


class FakeI2P:
    """A SAM v3 bridge that accepts one session and connects streams to known destinations."""

    def __init__(self, destinations: dict[str, tuple[str, int]]) -> None:
        self.destinations = destinations
        self.sessions: set[str] = set()
        self.commands: list[str] = []

    async def start(self) -> int:
        server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return int(server.sockets[0].getsockname()[1])

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hello = (await reader.readline()).decode()
        assert hello.startswith("HELLO VERSION")
        writer.write(b"HELLO REPLY RESULT=OK VERSION=3.1\n")
        command = (await reader.readline()).decode().strip()
        self.commands.append(command)
        fields = dict(part.split("=", 1) for part in command.split()[2:] if "=" in part)
        if command.startswith("SESSION CREATE"):
            self.sessions.add(fields["ID"])
            writer.write(b"SESSION STATUS RESULT=OK DESTINATION=transient\n")
            await writer.drain()
            await reader.read()  # the session lasts while this connection is open
            self.sessions.discard(fields["ID"])
            return
        assert command.startswith("STREAM CONNECT")
        if fields["ID"] not in self.sessions:
            writer.write(b"STREAM STATUS RESULT=INVALID_ID\n")
        elif fields["DESTINATION"] not in self.destinations:
            writer.write(b'STREAM STATUS RESULT=CANT_REACH_PEER MESSAGE="unknown"\n')
        else:
            upstream = await asyncio.open_connection(*self.destinations[fields["DESTINATION"]])
            writer.write(b"STREAM STATUS RESULT=OK\n")
            await writer.drain()
            await _splice(reader, writer, *upstream)
            return
        await writer.drain()
        writer.close()


async def _splice(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    other_reader: asyncio.StreamReader,
    other_writer: asyncio.StreamWriter,
) -> None:
    async def pump(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
        try:
            while data := await source.read(65536):
                sink.write(data)
                await sink.drain()
        finally:
            sink.close()

    await asyncio.gather(
        pump(reader, other_writer), pump(other_reader, writer), return_exceptions=True
    )


def _quic_echo_through(carrier_kind: str, mtls: MutualTLSIdentities) -> tuple[bytes, object]:
    """Send one real mutual-TLS QUIC request through a carrier; return the reply and the fake."""

    async def exercise() -> tuple[bytes, object]:
        target = xurcroh_quic.TensorServer.bind_mutual(
            "127.0.0.1:0",
            [mtls["server_certificate"], mtls["ca"]],
            mtls["server_key"],
            mtls["ca"],
        )
        tcp = await TcpDatagramRelay.bind("127.0.0.1", 0, *target.local_addr().rsplit(":", 1))
        name = "xurcrohworkerexample.onion" if carrier_kind == "tor" else "workerdest.b32.i2p"
        fake: FakeTor | FakeI2P
        if carrier_kind == "tor":
            fake = FakeTor({name: ("127.0.0.1", tcp.port)})
        else:
            fake = FakeI2P({name: ("127.0.0.1", tcp.port)})
        proxy_port = await fake.start()
        relay = await asyncio.to_thread(
            CarrierRelay, CarrierSpec(carrier_kind, name, 9446, "127.0.0.1", proxy_port)
        )

        def echo() -> None:
            connection = target.accept()
            values = connection.receive()
            connection.send(*values[:3], values[3], values[4] or "", values[5], values[6])
            connection.close()

        thread = threading.Thread(target=echo, daemon=True)
        thread.start()

        def request() -> bytes:
            client = xurcroh_quic.TensorClient.connect_mutual(
                relay.address,
                "localhost",
                mtls["ca"],
                [mtls["client_certificate"], mtls["ca"]],
                mtls["client_key"],
            )
            assert client.key_exchange_group() == "X25519MLKEM768"
            client.send(bytes(16), 0, 0, "tensor", "u8", [4], b"test")
            response = client.receive()
            client.close()
            return bytes(response[-1])

        try:
            return await asyncio.to_thread(request), fake
        finally:
            await asyncio.to_thread(relay.close)
            tcp.close()
            target.close()
            thread.join(timeout=5)

    return asyncio.run(exercise())


def test_real_quic_travels_through_a_tor_socks_proxy_to_an_onion_address(
    mtls_identities: MutualTLSIdentities,
) -> None:
    reply, fake = _quic_echo_through("tor", mtls_identities)

    assert reply == b"test"
    assert isinstance(fake, FakeTor)
    name, port, user, password = fake.requests[0]
    # The onion name reached the proxy as a name, with credentials for circuit isolation.
    assert (name, port) == ("xurcrohworkerexample.onion", 9446)
    assert user and password


def test_real_quic_travels_through_an_i2p_sam_bridge(
    mtls_identities: MutualTLSIdentities,
) -> None:
    reply, fake = _quic_echo_through("i2p", mtls_identities)

    assert reply == b"test"
    assert isinstance(fake, FakeI2P)
    assert fake.commands[0].startswith("SESSION CREATE STYLE=STREAM")
    assert "DESTINATION=workerdest.b32.i2p" in fake.commands[1]
    assert "TO_PORT=9446" in fake.commands[1]


def test_each_role_gets_its_own_tor_circuit_credentials() -> None:
    first, second = Socks5Dialer("127.0.0.1", 9050), Socks5Dialer("127.0.0.1", 9050)

    assert first._credentials != second._credentials
    assert Socks5Dialer("127.0.0.1", 9050, isolate=False)._credentials is None


def test_proxy_failures_are_reported_with_their_reason() -> None:
    async def exercise() -> None:
        refusing = FakeTor({}, status=5)
        port = await refusing.start()
        with pytest.raises(ProtocolError, match="connection refused"):
            await Socks5Dialer("127.0.0.1", port)("example.onion", 80)
        with pytest.raises(ProtocolError, match="valid host and port"):
            await Socks5Dialer("127.0.0.1", port)("example.onion", 0)

        bridge = FakeI2P({})
        sam_port = await bridge.start()
        dialer = SamDialer("127.0.0.1", sam_port)
        with pytest.raises(ProtocolError, match="unknown"):
            await dialer("nowhere.b32.i2p", 1)
        with pytest.raises(ProtocolError, match="valid I2P destination"):
            await dialer("has space", 1)
        dialer.close()

    asyncio.run(exercise())


def test_carrier_is_read_from_worker_labels() -> None:
    assert carrier_from_labels({}, worker="w") is None
    tor = carrier_from_labels({"carrier": "tor", "carrier_target": "abc.onion:9446"}, worker="w")
    assert tor == CarrierSpec("tor", "abc.onion", 9446, "127.0.0.1", 9050)
    i2p = carrier_from_labels(
        {"carrier": "i2p", "carrier_target": "dest.b32.i2p:1", "carrier_proxy": "10.0.0.2:7000"},
        worker="w",
    )
    assert i2p == CarrierSpec("i2p", "dest.b32.i2p", 1, "10.0.0.2", 7000)
    for labels in (
        {"carrier": "tor"},
        {"carrier": "smoke-signals", "carrier_target": "a:1"},
        {"carrier": "tor", "carrier_target": "no-port"},
        {"carrier": "tor", "carrier_target": "a:1", "carrier_proxy": "bad"},
        {"carrier_target": "a:1"},
    ):
        with pytest.raises(ConfigurationError):
            carrier_from_labels(labels, worker="w")


def test_public_sdk_runs_a_fortress_model_over_the_tor_carrier(
    tiny_llama_path, tmp_path, mtls_identities: MutualTLSIdentities
) -> None:
    """Four role workers reached only through a SOCKS5 proxy, by onion-style names."""

    from tests.fortress.test_sdk import _signed_remote_grid

    from Xurcroh import Xurcroh

    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, 2)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def on_loop(coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=30)

    names: dict[str, tuple[str, int]] = {}
    relays = []
    for index, entry in enumerate(cluster):
        host, port = entry["endpoint"].removeprefix("quic://").rsplit(":", 1)
        relay = on_loop(TcpDatagramRelay.bind("127.0.0.1", 0, host, port))
        relays.append(relay)
        names[f"role{index}xurcroh.onion"] = ("127.0.0.1", relay.port)
    tor = FakeTor(names)
    proxy_port = on_loop(tor.start())
    for index, entry in enumerate(cluster):
        # The client is told only the onion name; the real endpoint is unreachable.
        entry["endpoint"] = "quic://192.0.2.1:9"
        entry["labels"].update(
            carrier="tor",
            carrier_target=f"role{index}xurcroh.onion:9446",
            carrier_proxy=f"127.0.0.1:{proxy_port}",
        )
    model = {
        "path": str(tiny_llama_path),
        "revision": "a" * 40,
        "engine": "fortress",
        "fortress_seed": 731,
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
    finally:
        for worker in workers:
            worker.close()
        for relay in relays:
            loop.call_soon_threadsafe(relay.close)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    assert response.usage.completion_tokens == 1
    routes = trace["transport"]["routes"]
    assert {route["carrier"] for route in routes} == {"tor"}
    # TLS and the role session are end to end through the carrier.
    assert all(route["negotiated"]["key_exchange_group"] == "X25519MLKEM768" for route in routes)
    assert all(route["negotiated"]["session_bound_to_tls"] for route in routes)
    assert len(trace["transport"]["worker_checkpoints"]) == 4
    assert {name for name, *_ in tor.requests} == set(names)
    assert len({user for _, _, user, _ in tor.requests}) == 4  # one circuit identity per role


def test_carrier_and_onion_route_together_are_refused() -> None:
    from xurcroh.fortress.engine import FortressEngine

    engine = object.__new__(FortressEngine)
    engine._carriers = {}
    labels = {"carrier": "tor", "carrier_target": "a.onion:1", "onion_public": "00"}

    with pytest.raises(ConfigurationError, match="use one"):
        engine._carried_address("w", labels, "10.0.0.1:9444")
    assert engine._carried_address("w", {}, "10.0.0.1:9444") == "10.0.0.1:9444"
