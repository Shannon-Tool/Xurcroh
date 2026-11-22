from __future__ import annotations

import secrets
import threading

from xurcroh.errors import ModelLoadError
from xurcroh.kem import HybridKemKey


class StageIdentityBroker:
    def __init__(self) -> None:
        self._identities: dict[str, HybridKemKey] = {}
        self._lock = threading.Lock()

    def register(self, private_key: HybridKemKey) -> str:
        handle = secrets.token_urlsafe(32)
        with self._lock:
            self._identities[handle] = private_key
        return handle

    def resolve(self, handle: str) -> HybridKemKey:
        with self._lock:
            try:
                return self._identities[handle]
            except KeyError as error:
                raise ModelLoadError("stage identity handle is unknown or revoked") from error

    def revoke(self, handle: str) -> None:
        with self._lock:
            self._identities.pop(handle, None)


_STAGE_IDENTITIES = StageIdentityBroker()


def register_stage_identity(private_key: HybridKemKey) -> str:
    return _STAGE_IDENTITIES.register(private_key)


def resolve_stage_identity(handle: str) -> HybridKemKey:
    return _STAGE_IDENTITIES.resolve(handle)


def revoke_stage_identity(handle: str) -> None:
    _STAGE_IDENTITIES.revoke(handle)
