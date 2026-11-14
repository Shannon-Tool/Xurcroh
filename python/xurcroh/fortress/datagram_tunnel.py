from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_DATAGRAM_BYTES = 65507
QUEUE_CAPACITY = 256


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(4)
    size = int.from_bytes(header, "big")
    if size < 1 or size > MAX_DATAGRAM_BYTES:
        raise ValueError("invalid datagram frame size")
    return await reader.readexactly(size)


async def write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    if not payload or len(payload) > MAX_DATAGRAM_BYTES:
        raise ValueError("invalid datagram size")
    writer.write(len(payload).to_bytes(4, "big") + payload)
    await writer.drain()


@dataclass
class TargetProtocol(asyncio.DatagramProtocol):
    queue: asyncio.Queue[bytes]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.queue.put_nowait(data)


@dataclass
class ClientConnection:
    relay: UdpDatagramRelay
    address: tuple[str, int]
    datagrams: asyncio.Queue[bytes] = field(
        default_factory=lambda: asyncio.Queue(maxsize=QUEUE_CAPACITY)
    )
    task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self.run())

    async def run(self) -> None:
        reader, writer = await asyncio.open_connection(
            self.relay.target_host, self.relay.target_port
        )
        try:
            sender = asyncio.create_task(self.send(writer))
            receiver = asyncio.create_task(self.receive(reader))
            done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
        finally:
            writer.close()
            await writer.wait_closed()

    async def send(self, writer: asyncio.StreamWriter) -> None:
        while True:
            payload = await self.datagrams.get()
            await write_frame(writer, payload)

    async def receive(self, reader: asyncio.StreamReader) -> None:
        while True:
            payload = await read_frame(reader)
            self.relay.transport.sendto(payload, self.address)


class TcpDatagramRelay:
    def __init__(
        self,
        server: asyncio.AbstractServer,
        target_host: str,
        target_port: int,
    ) -> None:
        self.server = server
        self.target_host = target_host
        self.target_port = target_port
        self.port = int(server.sockets[0].getsockname()[1])  # type: ignore[attr-defined]

    @classmethod
    async def bind(
        cls, host: str, port: int, target_host: str, target_port: int
    ) -> TcpDatagramRelay:
        holder: TcpDatagramRelay | None = None

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            assert holder is not None
            await holder.handle(reader, writer)

        server = await asyncio.start_server(accept, host, port)
        holder = cls(server, target_host, target_port)
        return holder

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
        transport, _ = await loop.create_datagram_endpoint(
            lambda: TargetProtocol(queue),
            remote_addr=(self.target_host, self.target_port),
        )
        forwarding = asyncio.create_task(self.forward(queue, writer))
        try:
            while True:
                payload = await read_frame(reader)
                transport.sendto(payload)
        finally:
            transport.close()
            forwarding.cancel()
            writer.close()
            await writer.wait_closed()

    async def forward(self, queue: asyncio.Queue[bytes], writer: asyncio.StreamWriter) -> None:
        while True:
            payload = await queue.get()
            await write_frame(writer, payload)

    def close(self) -> None:
        self.server.close()


class UdpDatagramRelay:
    def __init__(
        self,
        transport: asyncio.DatagramTransport,
        target_host: str,
        target_port: int,
    ) -> None:
        self.transport = transport
        self.target_host = target_host
        self.target_port = target_port
        self.clients: dict[tuple[str, int], ClientConnection] = {}

    @classmethod
    async def bind(
        cls, host: str, port: int, target_host: str, target_port: int
    ) -> UdpDatagramRelay:
        holder: UdpDatagramRelay | None = None

        class Protocol(asyncio.DatagramProtocol):
            def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
                assert holder is not None
                holder.receive(data, addr)

        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(Protocol, local_addr=(host, port))
        holder = cls(transport, target_host, target_port)
        return holder

    @property
    def port(self) -> int:
        return int(self.transport.get_extra_info("sockname")[1])

    @property
    def address(self) -> str:
        host, port = self.transport.get_extra_info("sockname")[:2]
        return f"{host}:{port}"

    def receive(self, data: bytes, address: tuple[str, int]) -> None:
        client = self.clients.get(address)
        if client is None:
            client = ClientConnection(self, address)
            self.clients[address] = client
            client.start()
        client.datagrams.put_nowait(data)

    def close(self) -> None:
        for client in self.clients.values():
            if client.task is not None:
                client.task.cancel()
        self.clients.clear()
        self.transport.close()


async def serve(args: argparse.Namespace) -> None:
    relay: Any
    if args.mode == "server":
        relay = await TcpDatagramRelay.bind(
            args.listen_host, args.listen_port, args.target_host, args.target_port
        )
    else:
        relay = await UdpDatagramRelay.bind(
            args.listen_host, args.listen_port, args.target_host, args.target_port
        )
    if args.address_file is not None:
        args.address_file.write_text(f"{relay.port}\n")
    try:
        await asyncio.Event().wait()
    finally:
        relay.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward framed datagrams over a TCP relay")
    parser.add_argument("mode", choices=("server", "client"))
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", required=True, type=int)
    parser.add_argument("--address-file", type=Path)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
