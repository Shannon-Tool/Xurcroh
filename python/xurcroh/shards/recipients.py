from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.errors import ShardIntegrityError

ENVELOPE_VERSION = 2
ALGORITHM = "X25519-HKDF-SHA256-ChaCha20-Poly1305"


@dataclass(frozen=True)
class StageRecipient:
    stage_index: int
    public_key: bytes
    key_id: str

    def __post_init__(self) -> None:
        if self.stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        if len(self.public_key) != 32:
            raise ValueError("recipient public key must contain 32 bytes")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.key_id):
            raise ValueError("invalid recipient key identifier")

    @classmethod
    def from_private_key(
        cls,
        stage_index: int,
        private_key: X25519PrivateKey,
        *,
        key_id: str | None = None,
    ) -> StageRecipient:
        public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return cls(stage_index, public, key_id or f"stage-{stage_index}")


def _validate_manifest_digest(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise ShardIntegrityError("manifest digest must be hexadecimal") from error
    if len(digest) != 32:
        raise ShardIntegrityError("manifest digest must contain 32 bytes")
    return digest


def _aad(
    recipient: StageRecipient,
    *,
    manifest_digest: str,
    checkpoint_fingerprint: str,
) -> bytes:
    if not checkpoint_fingerprint:
        raise ShardIntegrityError("checkpoint fingerprint is required")
    _validate_manifest_digest(manifest_digest)
    return (
        f"xurcroh-stage-recipient-v2:{recipient.key_id}:{recipient.stage_index}:"
        f"{manifest_digest}:{checkpoint_fingerprint}"
    ).encode()


def _derive_key(
    shared: bytes,
    *,
    manifest_digest: str,
    aad: bytes,
) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=bytes.fromhex(manifest_digest),
        info=aad,
    ).derive(shared)


def seal_stage_key(
    stage_key: bytes,
    recipient: StageRecipient,
    *,
    manifest_digest: str,
    checkpoint_fingerprint: str,
) -> dict[str, str]:
    if len(stage_key) != 32:
        raise ValueError("stage key must contain 32 bytes")
    aad = _aad(
        recipient,
        manifest_digest=manifest_digest,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    ephemeral = X25519PrivateKey.generate()
    ephemeral_public = ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    shared = ephemeral.exchange(recipient_public_key(recipient))
    key = _derive_key(shared, manifest_digest=manifest_digest, aad=aad)
    nonce = os.urandom(12)
    wrapped = ChaCha20Poly1305(key).encrypt(nonce, stage_key, aad)
    return {
        "version": str(ENVELOPE_VERSION),
        "algorithm": ALGORITHM,
        "key_id": recipient.key_id,
        "stage_index": str(recipient.stage_index),
        "recipient_public": base64.b64encode(recipient.public_key).decode(),
        "ephemeral_public": base64.b64encode(ephemeral_public).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "wrapped_key": base64.b64encode(wrapped).decode(),
    }


def recipient_public_key(recipient: StageRecipient) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(recipient.public_key)


def open_stage_key(
    envelope: Mapping[str, str],
    private_key: X25519PrivateKey,
    *,
    manifest_digest: str,
    checkpoint_fingerprint: str,
) -> bytes:
    try:
        if envelope["version"] != str(ENVELOPE_VERSION) or envelope["algorithm"] != ALGORITHM:
            raise ShardIntegrityError("unsupported recipient key envelope")
        recipient = StageRecipient(
            stage_index=int(envelope["stage_index"]),
            public_key=base64.b64decode(envelope["recipient_public"], validate=True),
            key_id=envelope["key_id"],
        )
        actual_public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        if not hmac.compare_digest(actual_public, recipient.public_key):
            raise ShardIntegrityError("recipient private key does not match the envelope")
        aad = _aad(
            recipient,
            manifest_digest=manifest_digest,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )
        ephemeral_public = base64.b64decode(envelope["ephemeral_public"], validate=True)
        if len(ephemeral_public) != 32:
            raise ShardIntegrityError("invalid ephemeral public key")
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

        shared = private_key.exchange(X25519PublicKey.from_public_bytes(ephemeral_public))
        key = _derive_key(shared, manifest_digest=manifest_digest, aad=aad)
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        wrapped = base64.b64decode(envelope["wrapped_key"], validate=True)
        if len(nonce) != 12:
            raise ShardIntegrityError("invalid recipient envelope nonce")
        return ChaCha20Poly1305(key).decrypt(nonce, wrapped, aad)
    except (KeyError, TypeError, ValueError, binascii.Error, InvalidTag) as error:
        raise ShardIntegrityError("cannot authenticate recipient key envelope") from error


def manifest_digest(document: Mapping[str, object]) -> str:
    import json

    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_manifest_digest(document: Mapping[str, object]) -> str:
    expected = document.get("digest")
    if not isinstance(expected, str):
        raise ShardIntegrityError("manifest digest is missing")
    payload = dict(document)
    payload.pop("digest", None)
    actual = manifest_digest(payload)
    if not hmac.compare_digest(actual, expected):
        raise ShardIntegrityError("manifest digest mismatch")
    return actual
