from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from xurcroh.errors import AttestationError

CANARY = bytes(range(32))
STORAGE_AAD = b"xurcroh-fortress-storage-v1"


class StorageIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class DispersedShard:
    threshold: int
    slices: dict[int, bytes] = field(repr=False)
    key_shares: dict[int, bytes] = field(repr=False)
    nonce: bytes = field(repr=False)
    aad: bytes = field(repr=False)


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right, strict=True))


def _gf_multiply(left: int, right: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        left = ((left << 1) ^ (0x11D if left & 0x80 else 0)) & 0xFF
        right >>= 1
    return result


def _gf_power(value: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf_multiply(result, value)
        value = _gf_multiply(value, value)
        exponent >>= 1
    return result


def _gf_inverse(value: int) -> int:
    if value == 0:
        raise StorageIntegrityError("zero has no multiplicative inverse in GF(256)")
    return _gf_power(value, 254)


def _rs_encode(package: bytes, threshold: int, slice_count: int) -> dict[int, bytes]:
    slice_length = (len(package) + threshold - 1) // threshold
    package += bytes(slice_length * threshold - len(package))
    blocks = [
        package[index : index + slice_length] for index in range(0, len(package), slice_length)
    ]
    slices = {index: block for index, block in enumerate(blocks)}
    for parity_index in range(slice_count - threshold):
        coefficients = [_gf_power(parity_index + 1, column) for column in range(threshold)]
        parity = bytearray(slice_length)
        for column, block in enumerate(blocks):
            for row, value in enumerate(block):
                parity[row] ^= _gf_multiply(coefficients[column], value)
        slices[threshold + parity_index] = bytes(parity)
    return slices


def _rs_decode(slices: dict[int, bytes], threshold: int) -> bytes:
    rows = []
    for index, payload in slices.items():
        if index < threshold:
            row = [int(column == index) for column in range(threshold)]
        else:
            row = [_gf_power(index - threshold + 1, column) for column in range(threshold)]
        rows.append(row + list(payload))
    for column in range(threshold):
        pivot = next((row for row in range(column, threshold) if rows[row][column]), None)
        if pivot is None:
            raise StorageIntegrityError("Reed-Solomon matrix is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        inverse = _gf_inverse(rows[column][column])
        rows[column] = [_gf_multiply(inverse, value) for value in rows[column]]
        for row_index in range(threshold):
            if row_index == column:
                continue
            factor = rows[row_index][column]
            if factor:
                rows[row_index] = [
                    left ^ _gf_multiply(factor, right)
                    for left, right in zip(rows[row_index], rows[column], strict=True)
                ]
    return b"".join(bytes(row[threshold:]) for row in rows)


def _shamir_split(key: bytes, threshold: int, slice_count: int) -> dict[int, bytes]:
    if len(key) != 32:
        raise ValueError("data key must contain 32 bytes")
    shares = {index: bytearray(32) for index in range(slice_count)}
    for position, secret in enumerate(key):
        coefficients = [secret, *os.urandom(threshold - 1)]
        for index in range(slice_count):
            value = 0
            x = index + 1
            for power, coefficient in enumerate(coefficients):
                value ^= _gf_multiply(coefficient, _gf_power(x, power))
            shares[index][position] = value
    return {index: bytes(value) for index, value in shares.items()}


def _shamir_recover(shares: dict[int, bytes]) -> bytes:
    if any(len(share) != 32 for share in shares.values()):
        raise StorageIntegrityError("key share must contain 32 bytes")
    key = bytearray(32)
    for index, left in shares.items():
        numerator = 1
        denominator = 1
        for other in shares:
            if other == index:
                continue
            numerator = _gf_multiply(numerator, other + 1)
            denominator = _gf_multiply(denominator, (index + 1) ^ (other + 1))
        factor = _gf_inverse(denominator)
        for position in range(32):
            key[position] ^= _gf_multiply(left[position], _gf_multiply(numerator, factor))
    return bytes(key)


def disperse_shard(
    plaintext: bytes, *, threshold: int, slice_count: int, aad: bytes = STORAGE_AAD
) -> DispersedShard:
    if not 1 <= threshold <= slice_count <= 255:
        raise ValueError("storage requires 1 <= threshold <= slice_count <= 255")
    data_key = os.urandom(32)
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(data_key).encrypt(nonce, plaintext, aad)
    body = len(ciphertext).to_bytes(8, "little") + ciphertext + CANARY
    body += bytes(-len(body) % 32)
    aont_key = os.urandom(32)
    cipher = Cipher(algorithms.ChaCha20(aont_key, bytes(16)), mode=None)
    masked = cipher.encryptor().update(body)
    digest = hashlib.sha256(masked).digest()
    package = masked + _xor(aont_key, digest)
    key_shares = _shamir_split(data_key, threshold, slice_count)
    return DispersedShard(
        threshold=threshold,
        slices=_rs_encode(package, threshold, slice_count),
        key_shares=key_shares,
        nonce=nonce,
        aad=aad,
    )


def reconstruct_shard(
    bundle: DispersedShard,
    slices: dict[int, bytes],
    key_shares: dict[int, bytes],
    attest: Callable[[int], bool],
) -> bytes:
    if len(slices) < bundle.threshold or len(key_shares) < bundle.threshold:
        raise ValueError("threshold slices and key shares are required")
    selected = sorted(set(slices) & set(key_shares))[: bundle.threshold]
    if len(selected) < bundle.threshold:
        raise ValueError("threshold slices and key shares are required")
    if any(not attest(index) for index in selected):
        raise AttestationError("key holder attestation failed")
    selected_slices = {index: slices[index] for index in selected}
    package = _rs_decode(selected_slices, bundle.threshold)
    package_length = len(package) - len(package) % 32
    masked = package[: package_length - 32]
    difference = package[package_length - 32 : package_length]
    aont_key = _xor(difference, hashlib.sha256(masked).digest())
    cipher = Cipher(algorithms.ChaCha20(aont_key, bytes(16)), mode=None)
    body = cipher.decryptor().update(masked)
    if len(body) < 40:
        raise StorageIntegrityError("AONT body is truncated")
    ciphertext_length = int.from_bytes(body[:8], "little")
    canary_start = 8 + ciphertext_length
    if canary_start + 32 > len(body) or body[canary_start : canary_start + 32] != CANARY:
        raise StorageIntegrityError("AONT canary does not match")
    if any(body[canary_start + 32 :]):
        raise StorageIntegrityError("AONT padding is invalid")
    ciphertext = body[8:canary_start]
    key = _shamir_recover({index: key_shares[index] for index in selected})
    try:
        return ChaCha20Poly1305(key).decrypt(bundle.nonce, ciphertext, bundle.aad)
    except InvalidTag as error:
        raise StorageIntegrityError("shard authentication failed") from error
