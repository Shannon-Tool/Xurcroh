from __future__ import annotations

import time

import pytest

from xurcroh.controller.identity import Registration, sign_registration
from xurcroh.controller.registry import WorkerRegistry
from xurcroh.errors import AttestationError
from xurcroh.signatures import HybridSigningKey


def test_signed_registration_and_impersonation_rejection() -> None:
    private = HybridSigningKey.generate()
    public = private.public_key
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
    private = HybridSigningKey.generate()
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
            registration, sign_registration(private, registration), private.public_key
        )


def test_worker_public_key_cannot_change_for_an_existing_identity() -> None:
    first = HybridSigningKey.generate()
    second = HybridSigningKey.generate()
    registration = Registration(
        worker_id="worker-a",
        endpoint="https://worker-a.internal",
        models=["tiny"],
        modalities=["text"],
        timestamp=time.time(),
    )
    registry = WorkerRegistry()
    registry.register(registration, sign_registration(first, registration), first.public_key)

    with pytest.raises(AttestationError, match="identity"):
        registry.register(registration, sign_registration(second, registration), second.public_key)
