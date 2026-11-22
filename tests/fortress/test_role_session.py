from __future__ import annotations

import pytest

from xurcroh import kem
from xurcroh.errors import ProtocolError
from xurcroh.fortress.session import (
    ClientRoleSession,
    RoleSessionStore,
    split_data_packet,
)
from xurcroh.signatures import HybridSigningKey

ARTIFACTS = "ab" * 32


def _pair(
    *,
    client_binding: bytes = bytes(32),
    worker_binding: bytes = bytes(32),
    pin_identity: bool = True,
    pinned_artifacts: str | None = ARTIFACTS,
) -> tuple[ClientRoleSession, RoleSessionStore, kem.HybridKemKey, HybridSigningKey, bytes, bytes]:
    recipient = kem.HybridKemKey.generate()
    identity = HybridSigningKey.generate()
    store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS, identity=identity)
    client = ClientRoleSession(
        1,
        recipient.public_key,
        binding=client_binding,
        identity_public=identity.public_key if pin_identity else None,
        artifact_digest=pinned_artifacts,
    )
    hello = client.hello()
    session_id, reply = store.accept(hello, worker_binding)
    assert session_id == client.session_id
    return client, store, recipient, identity, hello, reply


def _exchange(client: ClientRoleSession, store: RoleSessionStore, text: bytes) -> bytes:
    """One request and its echo through the established channels."""

    session_id, sealed = split_data_packet(
        client.data_packet(client.sender.seal("r", "u8", [len(text)], text))
    )
    receiver, sender = store.channels(session_id)
    received = receiver.open(sealed)
    return client.receiver.open(sender.seal("r", "u8", [len(received)], received))


def test_handshake_establishes_matching_keys_and_reports_what_was_checked() -> None:
    client, store, *_, reply = _pair()
    client.finish(reply)

    assert client.established and client.worker_signed
    assert client.artifact_digest == ARTIFACTS
    assert client.tls_bound is False
    assert _exchange(client, store, b"activation") == b"activation"
    store.authorize(client.session_id, client.token_bytes)
    with pytest.raises(ProtocolError, match="not authorized"):
        store.authorize(client.session_id, bytes(32))


def test_recorded_session_cannot_be_opened_with_the_workers_static_key_alone() -> None:
    """Forward secrecy: the worker's long-term key does not recover the session keys."""

    client, store, recipient, _, hello, reply = _pair()
    client.finish(reply)
    recorded = client.sender.seal("r", "u8", [6], b"secret")

    # An attacker holding the recording and the static key replays the hello to a store
    # built from that key. It gets a new worker secret, never the recorded one.
    attacker_store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS)
    session_id, _ = attacker_store.accept(hello)
    receiver, _ = attacker_store.channels(session_id)
    with pytest.raises(ProtocolError):
        receiver.open(recorded)
    # The genuine worker still opens it.
    genuine_receiver, _ = store.channels(client.session_id)
    assert genuine_receiver.open(recorded) == b"secret"


def test_ratchet_keys_do_not_open_traffic_from_another_epoch() -> None:
    client, store, *_, reply = _pair()
    client.finish(reply)
    before = client.sender.seal("r", "u8", [3], b"old")
    client.rotate(1)
    after = client.sender.seal("r", "u8", [3], b"new")
    receiver, _ = store.channels(client.session_id)

    with pytest.raises(ProtocolError):
        receiver.open(after)  # worker still on the old key
    store.rotate(client.session_id, 1)
    assert receiver.open(after) == b"new"
    with pytest.raises(ProtocolError):
        receiver.open(before)  # the old key is gone
    assert client.sender._master_key == b""
    with pytest.raises(ProtocolError, match="epoch"):
        store.rotate(client.session_id, 3)
    with pytest.raises(ProtocolError, match="epoch"):
        client.rotate(0)


def test_session_bound_to_one_tls_connection_fails_on_another() -> None:
    client, store, *_, reply = _pair(client_binding=b"\x01" * 32, worker_binding=b"\x01" * 32)
    client.finish(reply)
    assert client.tls_bound is True
    assert _exchange(client, store, b"x") == b"x"

    # A hello made for one TLS connection is refused when it arrives over another.
    with pytest.raises(ProtocolError, match="does not open"):
        _pair(client_binding=b"\x01" * 32, worker_binding=b"\x02" * 32)


def test_reply_from_another_identity_or_for_other_artifacts_is_rejected() -> None:
    client, _, recipient, _, hello, _ = _pair()
    impostor = RoleSessionStore(
        recipient, 1, artifact_digest=ARTIFACTS, identity=HybridSigningKey.generate()
    )
    with pytest.raises(ProtocolError, match="pinned worker identity"):
        client.finish(impostor.accept(hello)[1])

    unsigned = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS)
    with pytest.raises(ProtocolError, match="pinned worker identity"):
        client.finish(unsigned.accept(hello)[1])

    other = _pair(pinned_artifacts="cd" * 32)
    with pytest.raises(ProtocolError, match="artifacts other than the ones deployed"):
        other[0].finish(other[-1])


def test_unpinned_client_accepts_but_reports_no_worker_signature() -> None:
    client, store, *_, reply = _pair(pin_identity=False, pinned_artifacts=None)
    client.finish(reply)

    assert client.worker_signed is False
    assert _exchange(client, store, b"x") == b"x"


@pytest.mark.parametrize("offset", [4, 40, 1300, -1])
def test_tampered_hello_is_rejected(offset: int) -> None:
    recipient = kem.HybridKemKey.generate()
    store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS)
    hello = bytearray(ClientRoleSession(1, recipient.public_key).hello())
    hello[offset] ^= 1

    with pytest.raises(ProtocolError):
        store.accept(bytes(hello))


@pytest.mark.parametrize("offset", [0, 1168, 1199, 1200, -1])
def test_tampered_reply_is_rejected(offset: int) -> None:
    client, *_, reply = _pair()
    damaged = bytearray(reply)
    damaged[offset] ^= 1

    with pytest.raises(ProtocolError):
        client.finish(bytes(damaged))


def test_hello_replay_and_wrong_role_are_rejected() -> None:
    _, store, recipient, _, hello, _ = _pair()

    with pytest.raises(ProtocolError, match="already in use"):
        store.accept(hello)
    other_role = RoleSessionStore(recipient, 2, artifact_digest=ARTIFACTS)
    with pytest.raises(ProtocolError, match="does not open"):
        other_role.accept(hello)


def test_expired_session_is_refused_and_a_new_handshake_works() -> None:
    recipient = kem.HybridKemKey.generate()
    store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS, ttl_seconds=0)
    first = ClientRoleSession(1, recipient.public_key)
    first.finish(store.accept(first.hello())[1])

    with pytest.raises(ProtocolError, match="unknown or has expired"):
        store.channels(first.session_id)

    store.ttl_seconds = 300
    second = ClientRoleSession(1, recipient.public_key)
    second.finish(store.accept(second.hello())[1])
    assert _exchange(second, store, b"again") == b"again"


def test_capacity_is_bounded_and_handshake_cannot_be_finished_twice() -> None:
    recipient = kem.HybridKemKey.generate()
    store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS, capacity=1)
    client = ClientRoleSession(1, recipient.public_key)
    reply = store.accept(client.hello())[1]
    client.finish(reply)

    with pytest.raises(ProtocolError, match="capacity"):
        store.accept(ClientRoleSession(1, recipient.public_key).hello())
    with pytest.raises(ProtocolError, match="already complete"):
        client.finish(reply)
    with pytest.raises(ProtocolError, match="already complete"):
        client.hello()
    with pytest.raises(ProtocolError, match="malformed"):
        split_data_packet(b"XFD1" + b"0" * 32)


def test_arbitrary_bytes_are_rejected_with_a_protocol_error_only() -> None:
    """A worker must never fail in an unexpected way on attacker-chosen input."""

    import random

    from xurcroh.fortress.onion import unwrap_onion

    generator = random.Random(20261122)
    recipient = kem.HybridKemKey.generate()
    store = RoleSessionStore(recipient, 1, artifact_digest=ARTIFACTS)
    genuine = ClientRoleSession(1, recipient.public_key).hello()
    lengths = [0, 1, 4, 35, 36, 37, 1251, len(genuine) - 1, len(genuine), len(genuine) + 1, 4096]
    for length in lengths:
        for prefix in (b"", b"XFH1", b"XFD1", genuine[:36], genuine[:1252]):
            body = prefix + generator.randbytes(max(0, length - len(prefix)))
            for parse in (store.accept, split_data_packet, lambda b: unwrap_onion(recipient, b)):
                with pytest.raises(ProtocolError):
                    parse(body)
    assert store.active_count == 0
