from __future__ import annotations

import itertools

import pytest

from xurcroh.errors import ProtocolError
from xurcroh.fortress.onion import LAYER_BLOCK_BYTES, seal_onion, unwrap_onion
from xurcroh.kem import SEAL_OVERHEAD_BYTES, HybridKemKey


def _path(count: int) -> tuple[list[HybridKemKey], list[tuple[bytes, str]]]:
    keys = [HybridKemKey.generate() for _ in range(count)]
    targets = [*(f"quic://hop-{index}:9445" for index in range(1, count)), ""]
    return keys, [(key.public_key, target) for key, target in zip(keys, targets, strict=True)]


def test_each_hop_learns_only_its_next_hop_and_the_last_recovers_the_payload() -> None:
    keys, hops = _path(3)
    packet = seal_onion(hops, b"inner frame")

    seen = []
    for key in keys:
        next_hop, packet = unwrap_onion(key, packet)
        seen.append(next_hop)

    assert seen == ["quic://hop-1:9445", "quic://hop-2:9445", ""]
    assert packet == b"inner frame"


def test_a_hop_cannot_open_a_layer_meant_for_another() -> None:
    keys, hops = _path(2)
    packet = seal_onion(hops, b"payload")

    with pytest.raises(ProtocolError, match="does not open"):
        unwrap_onion(keys[1], packet)
    _, inner = unwrap_onion(keys[0], packet)
    with pytest.raises(ProtocolError, match="does not open"):
        unwrap_onion(keys[0], inner)


def test_layer_sizes_depend_on_blocks_not_on_payload_or_address_length() -> None:
    keys, _ = _path(1)
    short = seal_onion([(keys[0].public_key, "quic://a:1")], b"x")
    longer = seal_onion(
        [(keys[0].public_key, "quic://a-much-longer-host.example:65535")], b"x" * 500
    )

    assert len(short) == len(longer) == SEAL_OVERHEAD_BYTES + LAYER_BLOCK_BYTES
    assert (len(seal_onion([(keys[0].public_key, "")], b"x" * 5000)) - SEAL_OVERHEAD_BYTES) % (
        LAYER_BLOCK_BYTES
    ) == 0


def test_a_fixed_frame_grows_by_a_predictable_amount_per_hop() -> None:
    frame = bytes(16_384)
    sizes = []
    for count in (1, 2, 3, 4):
        _, hops = _path(count)
        sizes.append(len(seal_onion(hops, frame)))

    growth = {later - earlier for earlier, later in itertools.pairwise(sizes)}
    # Every extra hop costs the same two blocks, whatever the route.
    assert growth == {2 * LAYER_BLOCK_BYTES}


@pytest.mark.parametrize("offset", [0, 1100, 1140, -1])
def test_tampered_layer_is_rejected(offset: int) -> None:
    keys, hops = _path(1)
    packet = bytearray(seal_onion(hops, b"payload"))
    packet[offset] ^= 1

    with pytest.raises(ProtocolError):
        unwrap_onion(keys[0], bytes(packet))


def test_malformed_input_is_rejected() -> None:
    keys, _ = _path(1)

    with pytest.raises(ProtocolError):
        unwrap_onion(keys[0], b"short")
    with pytest.raises(ValueError, match="at least one hop"):
        seal_onion([], b"x")
    with pytest.raises(ValueError, match="hop address"):
        seal_onion([(keys[0].public_key, "x" * 70_000)], b"x")
    with pytest.raises(ProtocolError, match="public key"):
        seal_onion([(bytes(32), "")], b"x")
