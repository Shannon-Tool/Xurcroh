from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from xurcroh.errors import ShardIntegrityError

ENVELOPE_VERSION = 1


def write_key_envelope(
    path: str | Path,
    stage_key: bytes,
    wrapping_key: bytes,
    *,
    key_id: str,
) -> None:
    if len(stage_key) != 32 or len(wrapping_key) != 32:
        raise ValueError("stage and wrapping keys must contain 32 bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nonce = os.urandom(12)
    aad = f"xurcroh-key-envelope-v1:{key_id}".encode()
    wrapped = ChaCha20Poly1305(wrapping_key).encrypt(nonce, stage_key, aad)
    document = {
        "version": ENVELOPE_VERSION,
        "algorithm": "ChaCha20-Poly1305",
        "key_id": key_id,
        "nonce": base64.b64encode(nonce).decode(),
        "wrapped_key": base64.b64encode(wrapped).decode(),
    }
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_key_envelope(
    path: str | Path,
    wrapping_key: bytes,
    *,
    expected_key_id: str | None = None,
) -> bytes:
    source = Path(path)
    try:
        document = json.loads(source.read_text())
        key_id = document["key_id"]
        if document["version"] != ENVELOPE_VERSION or document["algorithm"] != "ChaCha20-Poly1305":
            raise ShardIntegrityError("unsupported key envelope")
        if expected_key_id is not None and key_id != expected_key_id:
            raise ShardIntegrityError("unexpected key envelope identity")
        nonce = base64.b64decode(document["nonce"], validate=True)
        wrapped = base64.b64decode(document["wrapped_key"], validate=True)
        aad = f"xurcroh-key-envelope-v1:{key_id}".encode()
        return ChaCha20Poly1305(wrapping_key).decrypt(nonce, wrapped, aad)
    except (OSError, KeyError, ValueError, InvalidTag) as error:
        raise ShardIntegrityError(f"cannot open key envelope {source}") from error
