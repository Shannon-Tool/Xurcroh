from __future__ import annotations

import pytest

from xurcroh.errors import ProtocolError
from xurcroh.kem import HybridKemKey
from xurcroh.runtime.stage_sessions import ClientStageSession, StageSessionStore


def test_stage_session_directions_use_distinct_keys_for_the_same_sequence() -> None:
    private = HybridKemKey.generate()
    recipient_public = private.public_key
    client = ClientStageSession(0, recipient_public)
    store = StageSessionStore(private, 0)
    receiver, sender = store.get_or_create(
        client.session_id, client.client_public, client.session_token, client.handshake
    )

    request = client.sender.seal("shared-request", "u8", [4], b"data")
    response = sender.seal("shared-request", "u8", [4], b"data")

    assert request[24:36] == response[24:36]
    assert request[36:] != response[36:]
    assert receiver.open_tensor(request)[2] == b"data"
    assert client.receiver.open_tensor(response)[2] == b"data"
    with pytest.raises(ProtocolError, match="authentication"):
        receiver.open_tensor(response)
    with pytest.raises(ProtocolError, match="authentication"):
        client.receiver.open_tensor(request)
    store.close()


def test_native_session_envelope_carries_an_authentication_secret() -> None:
    private = HybridKemKey.generate()
    recipient_public = private.public_key
    client = ClientStageSession(0, recipient_public)

    packet = client.wire_packet(b"payload")

    # digest, session id, token, then the master sealed with ML-KEM-768 and X25519
    assert len(packet) == 32 + 32 + 32 + (1136 + 32) + len(b"payload")
    assert packet[32:64] == client.session_id.encode()
    assert packet[96 : 96 + 1168] == client.handshake
