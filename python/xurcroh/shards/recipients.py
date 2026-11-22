from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass

from xurcroh.errors import ShardIntegrityError
from xurcroh.kem import PUBLIC_KEY_BYTES, HybridKemKey, KemError, open_sealed, seal

ENVELOPE_VERSION = 3
ALGORITHM = "HPKE-MLKEM768-X25519-HKDF-SHA256-ChaCha20-Poly1305"


@dataclass(frozen=True)
class StageRecipient:
    stage_index: int
    public_key: bytes
    key_id: str

    def __post_init__(self) -> None:
        if self.stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        if len(self.public_key) != PUBLIC_KEY_BYTES:
            raise ValueError(f"recipient public key must contain {PUBLIC_KEY_BYTES} bytes")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.key_id):
            raise ValueError("invalid recipient key identifier")

    @classmethod
    def from_private_key(
        cls,
        stage_index: int,
        private_key: HybridKemKey,
        *,
        key_id: str | None = None,
    ) -> StageRecipient:
        return cls(stage_index, private_key.public_key, key_id or f"stage-{stage_index}")


def _validate_manifest_digest(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise ShardIntegrityError("manifest digest must be hexadecimal") from error
    if len(digest) != 32:
        raise ShardIntegrityError("manifest digest must contain 32 bytes")
    return digest


def _info(
    recipient: StageRecipient,
    *,
    manifest_digest: str,
    checkpoint_fingerprint: str,
) -> bytes:
    """What a sealed stage key is bound to: the recipient, stage, manifest and checkpoint."""

    if not checkpoint_fingerprint:
        raise ShardIntegrityError("checkpoint fingerprint is required")
    _validate_manifest_digest(manifest_digest)
    return (
        f"xurcroh-stage-recipient-v3:{recipient.key_id}:{recipient.stage_index}:"
        f"{manifest_digest}:{checkpoint_fingerprint}"
    ).encode()


def seal_stage_key(
    stage_key: bytes,
    recipient: StageRecipient,
    *,
    manifest_digest: str,
    checkpoint_fingerprint: str,
) -> dict[str, str]:
    """Seal a stage key to its recipient with ML-KEM-768 and X25519 together (RFC 9180)."""

    if len(stage_key) != 32:
        raise ValueError("stage key must contain 32 bytes")
    info = _info(
        recipient,
        manifest_digest=manifest_digest,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    return {
        "version": str(ENVELOPE_VERSION),
        "algorithm": ALGORITHM,
        "key_id": recipient.key_id,
        "stage_index": str(recipient.stage_index),
        "recipient_public": base64.b64encode(recipient.public_key).decode(),
        "sealed_key": base64.b64encode(seal(recipient.public_key, info, stage_key)).decode(),
    }


def open_stage_key(
    envelope: Mapping[str, str],
    private_key: HybridKemKey,
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
        if not hmac.compare_digest(private_key.public_key, recipient.public_key):
            raise ShardIntegrityError("recipient private key does not match the envelope")
        info = _info(
            recipient,
            manifest_digest=manifest_digest,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )
        stage_key = open_sealed(
            private_key, info, base64.b64decode(envelope["sealed_key"], validate=True)
        )
        if len(stage_key) != 32:
            raise ShardIntegrityError("sealed stage key has the wrong length")
        return stage_key
    except (KeyError, TypeError, ValueError, binascii.Error, KemError) as error:
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
