from __future__ import annotations

import socket
import struct

import pytest

from xurcroh.errors import ProviderError
from xurcroh.providers.nitro_relay import (
    MAX_DATAGRAM_SIZE,
    NitroParentRelay,
    decode_frames,
    encode_frame,
    receive_exact,
)


def test_frame_round_trip_handles_fragmentation() -> None:
    payload = b"opaque encrypted datagram"
    framed = encode_frame(payload)
    stream = b"".join(bytes([value]) for value in framed)
    assert decode_frames(stream) == ([payload], b"")
    assert decode_frames(framed[:3]) == ([], framed[:3])


def test_frame_rejects_oversized_datagram() -> None:
    with pytest.raises(ValueError, match="datagram"):
        encode_frame(bytes(MAX_DATAGRAM_SIZE + 1))
    with pytest.raises(ProviderError, match="frame"):
        decode_frames(struct.pack("!I", MAX_DATAGRAM_SIZE + 1))


def test_receive_exact_rejects_truncated_frame() -> None:
    first, second = socket.socketpair()
    try:
        first.sendall(b"abc")
        first.shutdown(socket.SHUT_WR)
        with pytest.raises(ProviderError, match="truncated"):
            receive_exact(second, 4)
    finally:
        first.close()
        second.close()


def test_parent_relays_only_opaque_datagrams_from_configured_peer() -> None:
    enclave, parent = socket.socketpair()
    peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    peer.bind(("127.0.0.1", 0))
    stranger.bind(("127.0.0.1", 0))
    peer.settimeout(0.05)
    relay = NitroParentRelay(
        bind_address=("127.0.0.1", 0),
        allowed_peer=peer.getsockname(),
        enclave_cid=16,
        enclave_port=5000,
        vsock_socket_factory=lambda: parent,
    )
    relay.start()
    assert relay.bound_address is not None
    try:
        stranger.sendto(b"drop", relay.bound_address)
        peer.sendto(b"ciphertext", relay.bound_address)
        for _ in range(4):
            relay.run_once(timeout=0.01)
        assert receive_exact(enclave, 4) == struct.pack("!I", len(b"ciphertext"))
        assert receive_exact(enclave, len(b"ciphertext")) == b"ciphertext"
        with pytest.raises(TimeoutError):
            peer.recvfrom(32)

        enclave.sendall(encode_frame(b"return ciphertext"))
        for _ in range(4):
            relay.run_once(timeout=0.01)
        returned, source = peer.recvfrom(64)
        assert returned == b"return ciphertext"
        assert source == relay.bound_address
    finally:
        relay.close()
        enclave.close()
        peer.close()
        stranger.close()
