from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.errors import ModelLoadError
from xurcroh.shards.key_broker import (
    register_stage_identity,
    resolve_stage_identity,
    revoke_stage_identity,
)


def test_stage_identity_broker_revokes_handles() -> None:
    private = X25519PrivateKey.generate()
    handle = register_stage_identity(private)
    assert resolve_stage_identity(handle) is private
    revoke_stage_identity(handle)
    with pytest.raises(ModelLoadError, match="revoked"):
        resolve_stage_identity(handle)
