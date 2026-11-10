from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.memory import lock_buffer, secure_zero, unlock_buffer

MAGIC = b"XURCROH\0"
FORMAT_VERSION = 1
PREFIX = struct.Struct("<8sHI12s")
MAX_HEADER_BYTES = 1_048_576


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes | bytearray) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_shard_path(path: str | Path) -> Path:
    shard_path = Path(path)
    if not shard_path.is_file():
        raise ShardIntegrityError(f"missing shard: {shard_path}")
    if shard_path.suffix != ".xurcroh":
        raise ShardIntegrityError(f"strict loading requires a .xurcroh shard: {shard_path}")
    return shard_path


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_shard(
    path: str | Path,
    tensors: dict[str, torch.Tensor],
    key: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if len(key) != 32:
        raise ValueError("shard encryption key must contain 32 bytes")
    destination = Path(path)
    if destination.suffix != ".xurcroh":
        raise ValueError("shard destination must use the .xurcroh suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {name: tensor.detach().clone().contiguous() for name, tensor in tensors.items()}
    plaintext = bytearray(save_tensors(contiguous))
    try:
        plaintext_hash = _digest(plaintext)
        shard_id = _digest(plaintext_hash.encode() + _canonical_json(metadata))
        header = _canonical_json(
            {
                "format": "xurcroh-shard",
                "version": FORMAT_VERSION,
                "payload_sha256": plaintext_hash,
                "shard_id": shard_id,
                "metadata": metadata,
            }
        )
        if len(header) > MAX_HEADER_BYTES:
            raise ValueError("shard metadata exceeds the one MiB header limit")
        nonce = os.urandom(12)
        prefix = PREFIX.pack(MAGIC, FORMAT_VERSION, len(header), nonce)
        ciphertext = ChaCha20Poly1305(key).encrypt(nonce, bytes(plaintext), prefix + header)
        ciphertext_hash = _digest(ciphertext)
        _atomic_write(destination, prefix + header + ciphertext)
        return {
            **metadata,
            "format_version": FORMAT_VERSION,
            "shard_id": shard_id,
            "payload_sha256": plaintext_hash,
            "ciphertext_sha256": ciphertext_hash,
        }
    finally:
        secure_zero(plaintext)


def read_shard(
    path: str | Path,
    key: bytes,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    shard_path = validate_shard_path(path)
    if len(key) != 32:
        raise ShardIntegrityError("shard key must contain 32 bytes")
    payload = shard_path.read_bytes()
    try:
        if len(payload) < PREFIX.size + 16:
            raise ShardIntegrityError("truncated shard header")
        magic, version, header_length, nonce = PREFIX.unpack_from(payload)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ShardIntegrityError("unsupported shard magic or format version")
        if header_length > MAX_HEADER_BYTES:
            raise ShardIntegrityError("shard header exceeds the size limit")
        header_start = PREFIX.size
        header_end = header_start + header_length
        if header_end + 16 > len(payload):
            raise ShardIntegrityError("truncated shard authentication tag")
        header = payload[header_start:header_end]
        ciphertext = payload[header_end:]
        document = json.loads(header)
        if document.get("format") != "xurcroh-shard" or document.get("version") != version:
            raise ShardIntegrityError("inconsistent shard header")
        metadata = document["metadata"]
        for field, expected in (expected_metadata or {}).items():
            if metadata.get(field) != expected:
                raise ShardIntegrityError(f"shard metadata mismatch for {field}")
        aad = payload[:header_end]
        plaintext = bytearray(ChaCha20Poly1305(key).decrypt(nonce, ciphertext, aad))
        try:
            lock_buffer(plaintext)
            if _digest(plaintext) != document["payload_sha256"]:
                raise ShardIntegrityError("decrypted shard digest mismatch")
            tensors = load_tensors(bytes(plaintext))
        finally:
            secure_zero(plaintext)
            unlock_buffer(plaintext)
        result = {
            **metadata,
            "format_version": version,
            "shard_id": document["shard_id"],
            "payload_sha256": document["payload_sha256"],
            "ciphertext_sha256": _digest(ciphertext),
        }
        return tensors, result
    except (InvalidTag, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ShardIntegrityError(f"cannot authenticate or parse shard {shard_path}") from error
