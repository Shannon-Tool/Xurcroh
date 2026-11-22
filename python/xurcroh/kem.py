"""Hybrid public-key sealing: X25519 together with ML-KEM-768 (FIPS 203).

Messages are sealed with HPKE (RFC 9180) using the combined ML-KEM-768/X25519
KEM, HKDF-SHA256 and ChaCha20-Poly1305, as implemented by the ``cryptography``
library. Recovering a sealed message requires breaking both key exchanges.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hpke
from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey, MLKEM768PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from xurcroh.errors import XurcrohError

ALGORITHM = "X25519+ML-KEM-768"
SUITE_NAME = "HPKE ML-KEM-768/X25519, HKDF-SHA256, ChaCha20-Poly1305"
_MLKEM_PUBLIC = 1184
_MLKEM_SEED = 64
_X25519 = 32
PUBLIC_KEY_BYTES = _MLKEM_PUBLIC + _X25519
PRIVATE_KEY_BYTES = _MLKEM_SEED + _X25519
# ML-KEM-768 ciphertext, X25519 ephemeral share, AEAD tag.
SEAL_OVERHEAD_BYTES = 1088 + 32 + 16

_SUITE = hpke.Suite(hpke.KEM.MLKEM768_X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.CHACHA20_POLY1305)


class KemError(XurcrohError):
    """A hybrid KEM key or sealed message is malformed or does not open."""


class HybridKemKey:
    def __init__(self, post_quantum: MLKEM768PrivateKey, classical: X25519PrivateKey) -> None:
        self._post_quantum = post_quantum
        self._classical = classical
        self._private = hpke.MLKEM768X25519PrivateKey(post_quantum, classical)

    @classmethod
    def generate(cls) -> HybridKemKey:
        return cls(MLKEM768PrivateKey.generate(), X25519PrivateKey.generate())

    @classmethod
    def from_bytes(cls, private_key: bytes) -> HybridKemKey:
        if len(private_key) != PRIVATE_KEY_BYTES:
            raise KemError(f"a hybrid KEM private key is {PRIVATE_KEY_BYTES} bytes")
        return cls(
            MLKEM768PrivateKey.from_seed_bytes(private_key[:_MLKEM_SEED]),
            X25519PrivateKey.from_private_bytes(private_key[_MLKEM_SEED:]),
        )

    def to_bytes(self) -> bytes:
        return self._post_quantum.private_bytes_raw() + self._classical.private_bytes_raw()

    @property
    def public_key(self) -> bytes:
        return (
            self._post_quantum.public_key().public_bytes_raw()
            + self._classical.public_key().public_bytes_raw()
        )


def _public(public_key: bytes) -> hpke.MLKEM768X25519PublicKey:
    if len(public_key) != PUBLIC_KEY_BYTES:
        raise KemError(f"a hybrid KEM public key is {PUBLIC_KEY_BYTES} bytes")
    try:
        return hpke.MLKEM768X25519PublicKey(
            MLKEM768PublicKey.from_public_bytes(public_key[:_MLKEM_PUBLIC]),
            X25519PublicKey.from_public_bytes(public_key[_MLKEM_PUBLIC:]),
        )
    except ValueError as error:
        raise KemError("hybrid KEM public key is invalid") from error


def seal(recipient_public: bytes, info: bytes, plaintext: bytes) -> bytes:
    """Seal ``plaintext`` to ``recipient_public``, bound to ``info``."""

    return _SUITE.encrypt(plaintext, _public(recipient_public), info)


def open_sealed(key: HybridKemKey, info: bytes, sealed: bytes) -> bytes:
    if len(sealed) < SEAL_OVERHEAD_BYTES:
        raise KemError("sealed message is truncated")
    try:
        return _SUITE.decrypt(sealed, key._private, info)
    except (InvalidTag, ValueError) as error:
        raise KemError("sealed message does not open") from error
