from __future__ import annotations

import selectors
import socket
import struct
import threading
from collections.abc import Callable

from xurcroh.errors import ProviderError

MAX_DATAGRAM_SIZE = 65_535


def encode_frame(datagram: bytes) -> bytes:
    if len(datagram) > MAX_DATAGRAM_SIZE:
        raise ValueError("datagram exceeds the AF_VSOCK relay limit")
    return struct.pack("!I", len(datagram)) + datagram


def decode_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    frames = []
    offset = 0
    while len(buffer) - offset >= 4:
        size = struct.unpack("!I", buffer[offset : offset + 4])[0]
        if size > MAX_DATAGRAM_SIZE:
            raise ProviderError("AF_VSOCK frame exceeds the datagram limit")
        end = offset + 4 + size
        if len(buffer) < end:
            break
        frames.append(buffer[offset + 4 : end])
        offset = end
    return frames, buffer[offset:]


def receive_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ProviderError("truncated AF_VSOCK frame")
        result.extend(chunk)
    return bytes(result)


class NitroParentRelay:
    def __init__(
        self,
        *,
        bind_address: tuple[str, int],
        allowed_peer: tuple[str, int],
        enclave_cid: int,
        enclave_port: int,
        vsock_socket_factory: Callable[[], socket.socket] | None = None,
    ) -> None:
        if enclave_cid < 3 or not 1 <= enclave_port <= 2**32 - 1:
            raise ValueError("a concrete enclave CID and valid port are required")
        if not bind_address[0] or not allowed_peer[0]:
            raise ValueError("relay bind and peer addresses must be explicit")
        self._bind_address = bind_address
        self._allowed_peer = allowed_peer
        self._enclave_address = (enclave_cid, enclave_port)
        self._factory = vsock_socket_factory
        self._udp: socket.socket | None = None
        self._vsock: socket.socket | None = None
        self._incoming = bytearray()
        self._outgoing = bytearray()
        self._stop = threading.Event()
        self.bound_address: tuple[str, int] | None = None

    def start(self) -> None:
        if self._udp is not None:
            raise ProviderError("Nitro parent relay is already running")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(self._bind_address)
            udp.setblocking(False)
            if self._factory is None:
                if not hasattr(socket, "AF_VSOCK"):
                    raise ProviderError("AF_VSOCK is unavailable on this host")
                vsock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
                vsock.connect(self._enclave_address)
            else:
                vsock = self._factory()
            vsock.setblocking(False)
        except Exception:
            udp.close()
            raise
        self._udp = udp
        self._vsock = vsock
        address = udp.getsockname()
        self.bound_address = (str(address[0]), int(address[1]))

    def run_once(self, *, timeout: float = 1.0) -> None:
        if self._udp is None or self._vsock is None:
            raise ProviderError("Nitro parent relay is not running")
        events = selectors.EVENT_READ
        if self._outgoing:
            events |= selectors.EVENT_WRITE
        with selectors.DefaultSelector() as selector:
            selector.register(self._udp, selectors.EVENT_READ, "udp")
            selector.register(self._vsock, events, "vsock")
            ready = selector.select(timeout)
        for key, mask in ready:
            if key.data == "udp":
                self._receive_udp()
            else:
                if mask & selectors.EVENT_WRITE:
                    self._write_vsock()
                if mask & selectors.EVENT_READ:
                    self._read_vsock()

    def _receive_udp(self) -> None:
        assert self._udp is not None
        try:
            datagram, peer = self._udp.recvfrom(MAX_DATAGRAM_SIZE)
        except BlockingIOError:
            return
        if peer != self._allowed_peer:
            return
        self._outgoing.extend(encode_frame(datagram))

    def _write_vsock(self) -> None:
        assert self._vsock is not None
        try:
            sent = self._vsock.send(self._outgoing)
        except BlockingIOError:
            return
        if sent == 0:
            raise ProviderError("enclave AF_VSOCK connection closed")
        del self._outgoing[:sent]

    def _read_vsock(self) -> None:
        assert self._vsock is not None and self._udp is not None
        try:
            chunk = self._vsock.recv(65_536)
        except BlockingIOError:
            return
        if not chunk:
            raise ProviderError("enclave AF_VSOCK connection closed")
        self._incoming.extend(chunk)
        frames, remainder = decode_frames(bytes(self._incoming))
        self._incoming = bytearray(remainder)
        for frame in frames:
            self._udp.sendto(frame, self._allowed_peer)

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self.run_once()
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self._udp is not None:
            self._udp.close()
            self._udp = None
        if self._vsock is not None:
            self._vsock.close()
            self._vsock = None
        self._incoming.clear()
        self._outgoing.clear()
        self.bound_address = None
