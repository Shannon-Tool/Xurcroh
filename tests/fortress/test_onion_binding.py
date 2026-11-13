from __future__ import annotations

import pytest
import xurcroh_quic
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def test_python_onion_layers_reveal_only_the_current_hop() -> None:
    relay_private = X25519PrivateKey.generate()
    worker_private = X25519PrivateKey.generate()

    def public(private: X25519PrivateKey) -> bytes:
        return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    layers = xurcroh_quic.seal_onion(
        [(public(relay_private), "quic://127.0.0.1:9000"), (public(worker_private), "")],
        bytes(range(32)),
        71,
        b"secret-payload",
    )
    next_hop, return_key, reply_id, remaining = xurcroh_quic.unwrap_onion(
        relay_private.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        layers[0],
    )
    assert next_hop == "quic://127.0.0.1:9000"
    assert return_key is None
    assert reply_id == 71
    assert remaining != b"secret-payload"
    next_hop, return_key, reply_id, payload = xurcroh_quic.unwrap_onion(
        worker_private.private_bytes(
            Encoding.Raw,
            PrivateFormat.Raw,
            NoEncryption(),
        ),
        remaining,
    )
    assert next_hop == ""
    assert return_key == bytes(range(32))
    assert reply_id == 71
    assert payload == b"secret-payload"
    with pytest.raises(ValueError):
        xurcroh_quic.unwrap_onion(
            relay_private.private_bytes(
                Encoding.Raw,
                PrivateFormat.Raw,
                NoEncryption(),
            ),
            remaining,
        )
