from __future__ import annotations

import base64

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError
from xurcroh.runtime.media_sessions import (
    ProtectedMediaSessionStore,
    encode_public,
    open_tensors,
    seal_tensors,
)


def test_media_session_authenticates_tensors_and_rejects_replay() -> None:
    client_private = X25519PrivateKey.generate()
    store = ProtectedMediaSessionStore()
    session_id, worker_public = store.create(encode_public(client_private.public_key()))
    shared = client_private.exchange(
        X25519PublicKey.from_public_bytes(base64.b64decode(worker_public))
    )
    sender = PipelineTensorInterceptor(shared, b"client", b"worker", session_id.encode())
    session = store.get(session_id)
    packet = seal_tensors({"value": torch.tensor([1, 2, 3])}, "request-1", sender)
    assert open_tensors(packet, "request-1", session.receiver)["value"].tolist() == [1, 2, 3]
    with pytest.raises(ProtocolError, match="replay"):
        open_tensors(packet, "request-1", session.receiver)
    with pytest.raises(ProtocolError):
        open_tensors(packet[:-2] + "AA", "request-1", session.receiver)
    store.remove(session_id)
    assert store.active_count == 0
    with pytest.raises(ProtocolError, match="expired"):
        store.get(session_id)


def test_media_session_capacity_fails_closed() -> None:
    store = ProtectedMediaSessionStore(capacity=1)
    private = X25519PrivateKey.generate()
    store.create(encode_public(private.public_key()))
    with pytest.raises(ProtocolError, match="capacity"):
        store.create(encode_public(private.public_key()))
    store.close()


def test_media_session_directions_use_distinct_keys_for_the_same_sequence() -> None:
    client_private = X25519PrivateKey.generate()
    store = ProtectedMediaSessionStore()
    session_id, worker_public = store.create(encode_public(client_private.public_key()))
    shared = client_private.exchange(
        X25519PublicKey.from_public_bytes(base64.b64decode(worker_public))
    )
    context = session_id.encode()
    client_sender = PipelineTensorInterceptor(shared, b"client", b"worker", context)
    client_receiver = PipelineTensorInterceptor(shared, b"client", b"worker", context)
    session = store.get(session_id)

    request = seal_tensors({"value": torch.tensor([1])}, "shared-request", client_sender)
    response = seal_tensors({"value": torch.tensor([1])}, "shared-request", session.sender)

    assert request != response
    assert open_tensors(request, "shared-request", session.receiver)["value"].item() == 1
    assert open_tensors(response, "shared-request", client_receiver)["value"].item() == 1
    with pytest.raises(ProtocolError, match="authentication"):
        open_tensors(response, "shared-request", session.receiver)
    with pytest.raises(ProtocolError, match="authentication"):
        open_tensors(request, "shared-request", client_receiver)
    store.close()
