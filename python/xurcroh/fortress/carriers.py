"""Carriers: how the client reaches a role worker's datagram relay.

QUIC runs over UDP, and Tor and I2P carry only streams. A carrier opens a stream
to the worker-side TCP relay (``xurcroh.fortress.datagram_tunnel``), and a local
relay turns the client's QUIC datagrams into frames on that stream. The QUIC
mutual TLS, the role session, fixed-size frames and cover traffic all stay end to
end inside the carrier.

- ``direct`` connects over TCP.
- ``tor`` connects through a Tor SOCKS5 proxy, to an onion service or a clearnet
  address. The hostname is sent to Tor unresolved, so nothing is looked up
  locally, and each role uses its own SOCKS credentials so Tor puts each on a
  separate circuit.
- ``i2p`` connects through an I2P SAM v3 bridge to an I2P destination.

A carrier hides the client's network address from the worker's host. It does not
make traffic unlinkable against an observer who watches both ends, and it is
slow: every model layer is a round trip through the anonymity network.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from xurcroh.errors import ConfigurationError, ProtocolError
from xurcroh.fortress.datagram_tunnel import UdpDatagramRelay

Stream = tuple[asyncio.StreamReader, asyncio.StreamWriter]
Dialer = Callable[[str, int], Awaitable[Stream]]
CARRIERS = ("direct", "tor", "i2p")
_SOCKS_ERRORS = {
    1: "general failure",
    2: "connection not allowed",
    3: "network unreachable",
    4: "host unreachable",
    5: "connection refused",
    6: "TTL expired",
    7: "command not supported",
    8: "address type not supported",
}


async def dial_direct(host: str, port: int) -> Stream:
    return await asyncio.open_connection(host, port)


class Socks5Dialer:
    """Open streams through a SOCKS5 proxy such as Tor's, without resolving names locally."""

    def __init__(self, proxy_host: str, proxy_port: int, *, isolate: bool = True) -> None:
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        # Tor separates streams that present different SOCKS credentials, so one
        # dialer per role keeps the roles on different circuits.
        self._credentials = (
            (secrets.token_hex(8).encode(), secrets.token_hex(8).encode()) if isolate else None
        )

    async def __call__(self, host: str, port: int) -> Stream:
        name = host.encode("idna") if not host.endswith(".onion") else host.encode("ascii")
        if not name or len(name) > 255 or not 0 < port < 65536:
            raise ProtocolError("carrier target is not a valid host and port")
        reader, writer = await asyncio.open_connection(self.proxy_host, self.proxy_port)
        try:
            method = 2 if self._credentials is not None else 0
            writer.write(bytes([5, 1, method]))
            await writer.drain()
            if await reader.readexactly(2) != bytes([5, method]):
                raise ProtocolError("SOCKS5 proxy refused the authentication method")
            if self._credentials is not None:
                user, password = self._credentials
                writer.write(bytes([1, len(user)]) + user + bytes([len(password)]) + password)
                await writer.drain()
                if await reader.readexactly(2) != b"\x01\x00":
                    raise ProtocolError("SOCKS5 proxy rejected the credentials")
            # Address type 3 sends the name itself; the proxy resolves it.
            writer.write(bytes([5, 1, 0, 3, len(name)]) + name + port.to_bytes(2, "big"))
            await writer.drain()
            version, status, _, kind = await reader.readexactly(4)
            if version != 5:
                raise ProtocolError("SOCKS5 proxy sent a malformed reply")
            if status != 0:
                reason = _SOCKS_ERRORS.get(status, f"error {status}")
                raise ProtocolError(f"SOCKS5 proxy could not connect: {reason}")
            if kind == 1:
                await reader.readexactly(4 + 2)
            elif kind == 4:
                await reader.readexactly(16 + 2)
            elif kind == 3:
                await reader.readexactly((await reader.readexactly(1))[0] + 2)
            else:
                raise ProtocolError("SOCKS5 proxy sent a malformed reply")
        except (asyncio.IncompleteReadError, ConnectionError) as error:
            writer.close()
            raise ProtocolError("SOCKS5 proxy closed the connection") from error
        except BaseException:
            writer.close()
            raise
        return reader, writer


def _sam_fields(line: bytes) -> dict[str, str]:
    fields = {}
    for part in line.decode("ascii", errors="replace").split()[2:]:
        name, separator, value = part.partition("=")
        if separator:
            fields[name] = value.strip('"')
    return fields


class SamDialer:
    """Open streams to I2P destinations through a SAM v3 bridge."""

    def __init__(self, sam_host: str, sam_port: int) -> None:
        self.sam_host = sam_host
        self.sam_port = sam_port
        self._session_id = f"xurcroh-{secrets.token_hex(8)}"
        self._control: Stream | None = None
        self._lock: asyncio.Lock | None = None

    async def _command(self, stream: Stream, command: str, expected: str) -> dict[str, str]:
        reader, writer = stream
        writer.write(command.encode("ascii") + b"\n")
        await writer.drain()
        line = await reader.readline()
        if not line.startswith(expected.encode()):
            raise ProtocolError("I2P SAM bridge sent an unexpected reply")
        fields = _sam_fields(line)
        if fields.get("RESULT", "OK") != "OK":
            detail = fields.get("MESSAGE") or fields.get("RESULT")
            raise ProtocolError(f"I2P SAM bridge refused the request: {detail}")
        return fields

    async def _hello(self) -> Stream:
        stream = await asyncio.open_connection(self.sam_host, self.sam_port)
        await self._command(stream, "HELLO VERSION MIN=3.1 MAX=3.3", "HELLO REPLY")
        return stream

    async def __call__(self, host: str, port: int) -> Stream:
        if not host or any(character.isspace() for character in host):
            raise ProtocolError("carrier target is not a valid I2P destination")
        if self._lock is None:
            self._lock = asyncio.Lock()
        try:
            async with self._lock:
                if self._control is None:
                    # The session lives as long as this control connection stays open.
                    control = await self._hello()
                    await self._command(
                        control,
                        f"SESSION CREATE STYLE=STREAM ID={self._session_id} "
                        "DESTINATION=TRANSIENT SIGNATURE_TYPE=7",
                        "SESSION STATUS",
                    )
                    self._control = control
            stream = await self._hello()
            await self._command(
                stream,
                f"STREAM CONNECT ID={self._session_id} DESTINATION={host} "
                f"TO_PORT={port} SILENT=false",
                "STREAM STATUS",
            )
        except (asyncio.IncompleteReadError, ConnectionError) as error:
            raise ProtocolError("I2P SAM bridge closed the connection") from error
        return stream

    def close(self) -> None:
        if self._control is not None:
            self._control[1].close()
            self._control = None


@dataclass(frozen=True)
class CarrierSpec:
    """A carrier for one role, as configured in its worker labels."""

    kind: str
    target_host: str
    target_port: int
    proxy_host: str
    proxy_port: int

    def dialer(self) -> Dialer:
        if self.kind == "tor":
            return Socks5Dialer(self.proxy_host, self.proxy_port)
        if self.kind == "i2p":
            return SamDialer(self.proxy_host, self.proxy_port)
        return dial_direct


def _host_port(value: str, what: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    try:
        number = int(port)
    except ValueError:
        number = 0
    if not separator or not host or not 0 < number < 65536:
        raise ConfigurationError(f"{what} must be HOST:PORT")
    return host, number


def carrier_from_labels(labels: dict[str, str], *, worker: str) -> CarrierSpec | None:
    """Read ``carrier``, ``carrier_target`` and ``carrier_proxy`` labels for a role.

    ``carrier_target`` is the worker-side TCP relay as the carrier names it: an onion
    address, an I2P destination, or a host. ``carrier_proxy`` is the local Tor SOCKS5
    port or I2P SAM bridge.
    """

    kind = labels.get("carrier")
    if kind is None or (kind == "direct" and "carrier_target" not in labels):
        if "carrier_target" in labels or "carrier_proxy" in labels:
            raise ConfigurationError(f"worker {worker} sets a carrier target without a carrier")
        return None
    if kind not in CARRIERS:
        raise ConfigurationError(f"worker {worker} has an unknown carrier {kind!r}")
    try:
        target = _host_port(labels["carrier_target"], "carrier_target")
    except KeyError as error:
        raise ConfigurationError(f"worker {worker} carrier needs a carrier_target") from error
    default = {"tor": "127.0.0.1:9050", "i2p": "127.0.0.1:7656", "direct": "127.0.0.1:1"}[kind]
    proxy = _host_port(labels.get("carrier_proxy", default), "carrier_proxy")
    return CarrierSpec(kind, target[0], target[1], proxy[0], proxy[1])


class CarrierRelay:
    """A local UDP endpoint whose datagrams travel to the worker over a carrier."""

    def __init__(self, spec: CarrierSpec) -> None:
        self.spec = spec
        self._dialer = spec.dialer()
        self._loop = asyncio.new_event_loop()
        self._relay: UdpDatagramRelay | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="xurcroh-carrier")
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._error is not None or self._relay is None:
            raise ProtocolError(f"carrier relay could not start: {self._error}")
        self.address: str = self._relay.address

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._relay = self._loop.run_until_complete(
                UdpDatagramRelay.bind(
                    "127.0.0.1",
                    0,
                    self.spec.target_host,
                    self.spec.target_port,
                    dial=self._dialer,
                )
            )
        except BaseException as error:
            self._error = error
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    @property
    def failures(self) -> list[str]:
        """Reasons carrier streams failed, for diagnostics."""

        return [] if self._relay is None else list(self._relay.failures)

    def close(self) -> None:
        async def shutdown() -> None:
            if self._relay is not None:
                self._relay.close()
            closer: Any = getattr(self._dialer, "close", None)
            if closer is not None:
                closer()
            # Let cancelled stream tasks finish before the loop stops.
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._loop.stop()

        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        self._thread.join(timeout=10)
        with contextlib.suppress(RuntimeError):
            self._loop.close()
