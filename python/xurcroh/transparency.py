from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xurcroh.signatures import HybridSigningKey, SignatureError, verify_hybrid

TRANSPARENCY_FORMAT = "xurcroh-transparency-v2"
TRANSPARENCY_CONTEXT = b"xurcroh-transparency-entry"


class TransparencyError(ValueError):
    pass


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()


def _entry_hash(entry: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry)).hexdigest()


class TransparencyLog:
    def __init__(self, signing_key: HybridSigningKey) -> None:
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
        entry["signature"] = self._signing_key.sign(TRANSPARENCY_CONTEXT, _canonical(entry)).hex()
        self.entries.append(entry)
        return dict(entry)

    def save(self, path: str | Path) -> None:
        document = {"format": TRANSPARENCY_FORMAT, "entries": self.entries}
        Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_transparency_log(
    document: dict[str, Any],
    *,
    public_key: bytes,
    expected_measurements: dict[str, str],
) -> None:
    if document.get("format") != TRANSPARENCY_FORMAT:
        raise TransparencyError("unsupported transparency-log format")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise TransparencyError("transparency entries are invalid")
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
            verify_hybrid(
                public_key, TRANSPARENCY_CONTEXT, _canonical(unsigned), bytes.fromhex(signature)
            )
        except (SignatureError, ValueError) as error:
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


def fetch_transparency_log(
    endpoint: str,
    *,
    public_key: bytes,
    expected_measurements: dict[str, str],
    client: Any | None = None,
) -> dict[str, Any]:
    from urllib.parse import urlsplit

    import httpx

    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise TransparencyError("remote transparency logs require an HTTPS endpoint")
    owns_client = client is None
    remote = client or httpx.Client(base_url=endpoint, timeout=30)
    try:
        response = remote.get("/v1/log")
        response.raise_for_status()
        document = response.json()
        if not isinstance(document, dict):
            raise TransparencyError("remote transparency document must be an object")
    except (httpx.HTTPError, ValueError) as error:
        raise TransparencyError("could not retrieve the remote transparency log") from error
    finally:
        if owns_client:
            remote.close()
    verify_transparency_log(
        document,
        public_key=public_key,
        expected_measurements=expected_measurements,
    )
    return document
