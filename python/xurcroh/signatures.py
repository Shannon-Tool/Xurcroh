"""Hybrid signatures: Ed25519 together with ML-DSA-65 (FIPS 204).

A signature is accepted only when both halves verify, so forging one requires
breaking both schemes. Every signature is bound to a caller-chosen context label,
so a signature made for one purpose is not valid for another.
"""

from __future__ import annotations

import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA65PrivateKey, MLDSA65PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from xurcroh.errors import XurcrohError

ALGORITHM = "Ed25519+ML-DSA-65"
_ED25519_PUBLIC = 32
_ED25519_SIGNATURE = 64
_MLDSA_PUBLIC = 1952
_MLDSA_SIGNATURE = 3309
_SEED = 32
PRIVATE_KEY_BYTES = 2 * _SEED
PUBLIC_KEY_BYTES = _ED25519_PUBLIC + _MLDSA_PUBLIC
SIGNATURE_BYTES = _ED25519_SIGNATURE + _MLDSA_SIGNATURE
_DOMAIN = b"xurcroh-hybrid-signature-v1"


class SignatureError(XurcrohError):
    """A hybrid signature, or a key for one, is malformed or does not verify."""


def _signed_bytes(context: bytes, message: bytes) -> bytes:
    # Length-prefixing the context keeps (context, message) pairs unambiguous.
    return _DOMAIN + len(context).to_bytes(4, "big") + context + message


def key_fingerprint(public_key: bytes) -> str:
    """Short stable name for a hybrid public key."""

    return hashlib.sha256(public_key).hexdigest()


class HybridSigningKey:
    def __init__(self, classical: Ed25519PrivateKey, post_quantum: MLDSA65PrivateKey) -> None:
        self._classical = classical
        self._post_quantum = post_quantum

    @classmethod
    def generate(cls) -> HybridSigningKey:
        return cls(Ed25519PrivateKey.generate(), MLDSA65PrivateKey.generate())

    @classmethod
    def from_bytes(cls, private_key: bytes) -> HybridSigningKey:
        if len(private_key) != PRIVATE_KEY_BYTES:
            raise SignatureError(f"a hybrid signing key is {PRIVATE_KEY_BYTES} bytes")
        return cls(
            Ed25519PrivateKey.from_private_bytes(private_key[:_SEED]),
            MLDSA65PrivateKey.from_seed_bytes(private_key[_SEED:]),
        )

    def to_bytes(self) -> bytes:
        classical = self._classical.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        return classical + self._post_quantum.private_bytes_raw()

    @property
    def public_key(self) -> bytes:
        classical = self._classical.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return classical + self._post_quantum.public_key().public_bytes_raw()

    def sign(self, context: bytes, message: bytes) -> bytes:
        signed = _signed_bytes(context, message)
        return self._classical.sign(signed) + self._post_quantum.sign(signed)


def verify_hybrid(public_key: bytes, context: bytes, message: bytes, signature: bytes) -> None:
    """Raise ``SignatureError`` unless both halves of ``signature`` verify."""

    if len(public_key) != PUBLIC_KEY_BYTES:
        raise SignatureError(f"a hybrid public key is {PUBLIC_KEY_BYTES} bytes")
    if len(signature) != SIGNATURE_BYTES:
        raise SignatureError(f"a hybrid signature is {SIGNATURE_BYTES} bytes")
    signed = _signed_bytes(context, message)
    try:
        Ed25519PublicKey.from_public_bytes(public_key[:_ED25519_PUBLIC]).verify(
            signature[:_ED25519_SIGNATURE], signed
        )
        MLDSA65PublicKey.from_public_bytes(public_key[_ED25519_PUBLIC:]).verify(
            signature[_ED25519_SIGNATURE:], signed
        )
    except (InvalidSignature, ValueError) as error:
        raise SignatureError("hybrid signature does not verify") from error
