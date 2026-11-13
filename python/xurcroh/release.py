from __future__ import annotations

import hashlib
import json
import tomllib
import uuid
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from xurcroh.transparency import TransparencyError


def _canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_sbom(project_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    components: list[dict[str, str]] = []
    requirements = root / "requirements.lock"
    if requirements.is_file():
        for line in requirements.read_text().splitlines():
            if "==" not in line or line.lstrip().startswith("#"):
                continue
            name, remainder = line.split("==", 1)
            version = remainder.split()[0]
            components.append({"type": "library", "name": name.strip(), "version": version})
    cargo = root / "Cargo.lock"
    if cargo.is_file():
        document = tomllib.loads(cargo.read_text())
        for package in document.get("package", []):
            components.append(
                {"type": "library", "name": package["name"], "version": package["version"]}
            )
    npm = root / "package-lock.json"
    if npm.is_file():
        document = json.loads(npm.read_text())
        for path, package in document.get("packages", {}).items():
            if path and "version" in package:
                components.append(
                    {
                        "type": "library",
                        "name": path.removeprefix("node_modules/"),
                        "version": package["version"],
                    }
                )
    components.sort(key=lambda component: (component["name"], component["version"]))
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "components": components,
    }
    Path(output).write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n")
    return sbom


def sign_release(
    artifact_directory: str | Path,
    *,
    sbom_path: str | Path,
    signing_key: Any,
) -> Path:
    root = Path(artifact_directory)
    destination = root / "release-manifest.json"
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": _digest(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != destination
    ]
    if not artifacts:
        raise ValueError("release directory contains no artifacts")
    manifest = {
        "format": "xurcroh-release-v1",
        "artifacts": artifacts,
        "sbom_sha256": _digest(Path(sbom_path)),
    }
    manifest["signature"] = signing_key.sign(_canonical(manifest)).hex()
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def verify_release(manifest_path: str | Path, *, public_key: Ed25519PublicKey) -> None:
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    if document.get("format") != "xurcroh-release-v1":
        raise TransparencyError("unsupported release-manifest format")
    signature = document.pop("signature", None)
    if not isinstance(signature, str):
        raise TransparencyError("release signature is missing")
    try:
        public_key.verify(bytes.fromhex(signature), _canonical(document))
    except (InvalidSignature, ValueError) as error:
        raise TransparencyError("release signature is invalid") from error
    for artifact in document["artifacts"]:
        artifact_path = path.parent / artifact["path"]
        if not artifact_path.is_file() or _digest(artifact_path) != artifact["sha256"]:
            raise TransparencyError(f"release artifact digest mismatch: {artifact['path']}")
