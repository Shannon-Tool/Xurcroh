from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.registry import WorkerRegistry
from xurcroh.errors import AttestationError


def test_signed_registration_and_impersonation_rejection() -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    registration = Registration(
        worker_id="worker-a",
        endpoint="https://worker-a.internal:9443",
        models=["qwen-7b"],
        modalities=["text"],
        timestamp=time.time(),
    )
    signature = sign_registration(private, registration)
    registry = WorkerRegistry(registration_ttl_seconds=30)
    registry.register(registration, signature, public)
    assert registry.get("worker-a").models == ("qwen-7b",)

    with pytest.raises(AttestationError, match="signature"):
        registry.register(registration, b"invalid", public)


def test_stale_registration_is_rejected() -> None:
    private = Ed25519PrivateKey.generate()
    registration = Registration(
        worker_id="worker-old",
        endpoint="https://worker-old.internal",
        models=[],
        modalities=["text"],
        timestamp=time.time() - 120,
    )
    registry = WorkerRegistry(registration_ttl_seconds=1)
    with pytest.raises(AttestationError, match="fresh"):
        registry.register(
            registration, sign_registration(private, registration), private.public_key()
        )
