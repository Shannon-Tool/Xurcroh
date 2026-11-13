from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class TransparencyError(ValueError):
    pass


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()


def _entry_hash(entry: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


class TransparencyLog:
    def __init__(self, signing_key: Ed25519PrivateKey) -> None:
        self._signing_key = signing_key
        self.entries: list[dict[str, Any]] = []

    def append(self, image_digest: str, measurement: str) -> dict[str, Any]:
        previous_hash = _entry_hash(self.entries[-1]) if self.entries else "0" * 64
        entry = {
            "sequence": len(self.entries),
            "recorded_at": datetime.now(UTC).isoformat(),
            "image_digest": image_digest,
            "measurement": measurement,
            "previous_hash": previous_hash,
        }
        entry["signature"] = self._signing_key.sign(_canonical(entry)).hex()
        self.entries.append(entry)
        return dict(entry)

    def save(self, path: str | Path) -> None:
        document = {"format": "xurcroh-transparency-v1", "entries": self.entries}
        Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_transparency_log(
    document: dict[str, Any],
    *,
    public_key: bytes,
    expected_measurements: dict[str, str],
) -> None:
    if document.get("format") != "xurcroh-transparency-v1":
        raise TransparencyError("unsupported transparency-log format")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise TransparencyError("transparency entries are invalid")
    verifier = Ed25519PublicKey.from_public_bytes(public_key)
    previous_hash = "0" * 64
    by_image: dict[str, str] = {}
    for sequence, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("sequence") != sequence:
            raise TransparencyError("transparency sequence is invalid")
        if entry.get("previous_hash") != previous_hash:
            raise TransparencyError("transparency hash chain is invalid")
        signature = entry.get("signature")
        if not isinstance(signature, str):
            raise TransparencyError("transparency signature is missing")
        unsigned = {key: value for key, value in entry.items() if key != "signature"}
        try:
            verifier.verify(bytes.fromhex(signature), _canonical(unsigned))
        except (InvalidSignature, ValueError) as error:
            raise TransparencyError("transparency signature is invalid") from error
        image = entry.get("image_digest")
        measurement = entry.get("measurement")
        if not isinstance(image, str) or not isinstance(measurement, str):
            raise TransparencyError("transparency measurement is invalid")
        by_image[image] = measurement
        previous_hash = _entry_hash(entry)
    for image, expected in expected_measurements.items():
        if by_image.get(image) != expected:
            raise TransparencyError(f"worker image measurement mismatch for {image}")
