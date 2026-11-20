from __future__ import annotations

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.errors import AttestationError
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker import FortressRoleWorker


def _artifact(weight_index: int = 0) -> tuple[bytes, bytes]:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return (
        encode_role_artifact(operation, weight_index, public),
        private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
    )


def _worker_kwargs(mtls_identities: MutualTLSIdentities) -> dict:
    return {
        "certificates": (mtls_identities["server_certificate"], mtls_identities["ca"]),
        "private_key": mtls_identities["server_key"],
        "client_ca": mtls_identities["ca"],
    }


def test_worker_role_artifact_decode_fails_closed_without_attestation(
    mtls_identities: MutualTLSIdentities,
) -> None:
    artifact, recipient_key = _artifact()
    with pytest.raises(ValueError, match="attestation"):
        FortressRoleWorker(
            {"operation": artifact}, recipient_key, **_worker_kwargs(mtls_identities)
        )


def test_worker_role_artifact_decode_enforces_attestation_policy(
    mtls_identities: MutualTLSIdentities,
) -> None:
    artifact, recipient_key = _artifact()
    kwargs = _worker_kwargs(mtls_identities)
    verified: list[int] = []
    worker = FortressRoleWorker(
        {"operation": artifact},
        recipient_key,
        attest=lambda index: verified.append(index) is None,
        **kwargs,
    )
    try:
        assert verified
    finally:
        worker.close()
    with pytest.raises(AttestationError):
        FortressRoleWorker(
            {"operation": artifact},
            recipient_key,
            attest=lambda index: False,
            **kwargs,
        )
    worker = FortressRoleWorker(
        {"operation": artifact}, recipient_key, allow_unattested=True, **kwargs
    )
    worker.close()
    with pytest.raises(ValueError, match="allow_unattested"):
        FortressRoleWorker(
            {"operation": artifact},
            recipient_key,
            attest=lambda index: True,
            allow_unattested=True,
            **kwargs,
        )
