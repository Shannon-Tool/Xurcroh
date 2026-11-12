from __future__ import annotations

import secrets
import threading

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.errors import ModelLoadError


class StageIdentityBroker:
    def __init__(self) -> None:
        self._identities: dict[str, X25519PrivateKey] = {}
        self._lock = threading.Lock()

    def register(self, private_key: X25519PrivateKey) -> str:
        handle = secrets.token_urlsafe(32)
        with self._lock:
            self._identities[handle] = private_key
        return handle

    def resolve(self, handle: str) -> X25519PrivateKey:
        with self._lock:
            try:
                return self._identities[handle]
            except KeyError as error:
                raise ModelLoadError("stage identity handle is unknown or revoked") from error

    def revoke(self, handle: str) -> None:
        with self._lock:
            self._identities.pop(handle, None)


_STAGE_IDENTITIES = StageIdentityBroker()


def register_stage_identity(private_key: X25519PrivateKey) -> str:
    return _STAGE_IDENTITIES.register(private_key)


def resolve_stage_identity(handle: str) -> X25519PrivateKey:
    return _STAGE_IDENTITIES.resolve(handle)


def revoke_stage_identity(handle: str) -> None:
    _STAGE_IDENTITIES.revoke(handle)
