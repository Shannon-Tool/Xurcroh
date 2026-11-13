from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from xurcroh.release import generate_sbom, sign_release, verify_release
from xurcroh.transparency import TransparencyError


def test_release_sbom_and_signature_cover_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "requirements.lock").write_text("torch==2.8.0 --hash=sha256:test\n")
    (root / "Cargo.lock").write_text('[[package]]\nname = "xurcroh-crypto"\nversion = "0.1.0"\n')
    (root / "package-lock.json").write_text(
        json.dumps({"packages": {"node_modules/react": {"version": "19.0.0"}}})
    )
    sbom = generate_sbom(root, tmp_path / "sbom.json")
    assert {component["name"] for component in sbom["components"]} == {
        "torch",
        "xurcroh-crypto",
        "react",
    }
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "xurcroh.whl").write_bytes(b"wheel-content")
    private = Ed25519PrivateKey.generate()
    manifest = sign_release(artifacts, sbom_path=tmp_path / "sbom.json", signing_key=private)
    verify_release(manifest, public_key=private.public_key())
    document = json.loads(manifest.read_text())
    document["artifacts"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(TransparencyError):
        verify_release(manifest, public_key=private.public_key())
