"""Onion layers for relayed Fortress traffic.

Each layer is sealed to one hop with X25519 and ML-KEM-768 together
(``xurcroh.kem``) and names only that hop's next address. Layers are padded to
whole blocks, so a layer's size does not reveal the address it carries.

Only requests are layered. Onion layers do not hide timing or volume and do not
detect a replayed layer; the session inside does that.
"""

from __future__ import annotations

from collections.abc import Sequence

from xurcroh import kem
from xurcroh.errors import ProtocolError

LAYER_BLOCK_BYTES = 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_INFO = b"xurcroh-onion-v3"
_END = 0x80


def _encode(next_hop: str, payload: bytes) -> bytes:
    address = next_hop.encode()
    if len(address) > 0xFFFF:
        raise ValueError("onion hop address is too long")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("onion payload is too large")
    body = (
        len(address).to_bytes(2, "little")
        + address
        + len(payload).to_bytes(8, "little")
        + payload
        + bytes([_END])
    )
    return body + bytes(-len(body) % LAYER_BLOCK_BYTES)


def _decode(plaintext: bytes) -> tuple[str, bytes]:
    if not plaintext or len(plaintext) % LAYER_BLOCK_BYTES or len(plaintext) < 11:
        raise ProtocolError("onion layer is malformed")
    address_length = int.from_bytes(plaintext[:2], "little")
    offset = 2 + address_length
    if len(plaintext) < offset + 9:
        raise ProtocolError("onion layer is malformed")
    try:
        next_hop = plaintext[2:offset].decode()
    except UnicodeDecodeError as error:
        raise ProtocolError("onion layer is malformed") from error
    payload_length = int.from_bytes(plaintext[offset : offset + 8], "little")
    start = offset + 8
    end = start + payload_length
    if (
        payload_length > MAX_PAYLOAD_BYTES
        or end >= len(plaintext)
        or plaintext[end] != _END
        or any(plaintext[end + 1 :])
    ):
        raise ProtocolError("onion layer is malformed")
    return next_hop, plaintext[start:end]


def seal_onion(hops: Sequence[tuple[bytes, str]], payload: bytes) -> bytes:
    """Wrap ``payload`` for a path of ``(hop public key, address after that hop)`` pairs.

    The first pair is the entry relay. The last is the destination, whose next
    address is empty.
    """

    if not hops:
        raise ValueError("an onion path needs at least one hop")
    packet = payload
    for public_key, next_hop in reversed(hops):
        try:
            packet = kem.seal(public_key, _INFO, _encode(next_hop, packet))
        except kem.KemError as error:
            raise ProtocolError(f"onion hop public key is invalid: {error}") from error
    return packet


def unwrap_onion(key: kem.HybridKemKey, layer: bytes) -> tuple[str, bytes]:
    """Open this hop's layer; return the next address and the remaining packet."""

    try:
        plaintext = kem.open_sealed(key, _INFO, layer)
    except kem.KemError as error:
        raise ProtocolError("onion layer does not open with this hop's key") from error
    return _decode(plaintext)
