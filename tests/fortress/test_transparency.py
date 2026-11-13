from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.transparency import TransparencyError, TransparencyLog, verify_transparency_log


def test_transparency_log_verifies_signed_measurement_chain_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    log = TransparencyLog(private)
    log.append("sha256:image-a", "sha256:measurement-a")
    log.append("sha256:image-b", "sha256:measurement-b")
    path = tmp_path / "transparency.json"
    log.save(path)
    document = json.loads(path.read_text())
    verify_transparency_log(
        document,
        public_key=public,
        expected_measurements={"sha256:image-b": "sha256:measurement-b"},
    )
    document["entries"][1]["measurement"] = "sha256:attacker"
    with pytest.raises(TransparencyError):
        verify_transparency_log(
            document,
            public_key=public,
            expected_measurements={"sha256:image-b": "sha256:measurement-b"},
        )
    document = json.loads(path.read_text())
    with pytest.raises(TransparencyError, match="measurement"):
        verify_transparency_log(
            document,
            public_key=public,
            expected_measurements={"sha256:image-b": "sha256:other"},
        )
