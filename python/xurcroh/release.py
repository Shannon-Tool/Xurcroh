from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import tomllib
import uuid
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from xurcroh.transparency import TransparencyError


def _canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_workspace(value: Any, project_root: str) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_workspace(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_workspace(item, project_root) for item in value]
    if isinstance(value, str):
        return value.replace(project_root, "/workspace")
    return value


def normalize_wheel_sbom(wheel_path: str | Path, *, project_root: str | Path) -> None:
    wheel = Path(wheel_path)
    root = str(Path(project_root).resolve())
    with ZipFile(wheel) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    payloads = {info.filename: payload for info, payload in entries}
    for info, payload in entries:
        if not info.filename.endswith(".cyclonedx.json"):
            continue
        document = json.loads(payload)
        normalized = _normalize_workspace(document, root)
        payloads[info.filename] = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode()
    record_names = [
        info.filename for info, _ in entries if info.filename.endswith(".dist-info/RECORD")
    ]
    if len(record_names) != 1:
        raise ValueError("native wheel must contain exactly one RECORD file")
    record_name = record_names[0]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for info, _ in entries:
        name = info.filename
        if name == record_name:
            writer.writerow((name, "", ""))
            continue
        payload = payloads[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((name, f"sha256={digest.decode()}", len(payload)))
    payloads[record_name] = output.getvalue().encode()
    temporary = wheel.with_suffix(wheel.suffix + ".tmp")
    try:
        with ZipFile(temporary, "w") as destination:
            for info, _ in entries:
                info.compress_type = ZIP_STORED
                executable = (info.external_attr >> 16) & 0o111
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                destination.writestr(info, payloads[info.filename])
        with ZipFile(temporary) as verification:
            if verification.testzip() is not None:
                raise ValueError("normalized native wheel failed ZIP integrity verification")
        temporary.replace(wheel)
    finally:
        temporary.unlink(missing_ok=True)


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
    inventory = json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"xurcroh:sbom:{hashlib.sha256(inventory).hexdigest()}")
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
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
        "format": "xurcroh-release-v2",
        "artifacts": artifacts,
        "sbom_path": str(Path(sbom_path).resolve()),
        "sbom_sha256": _digest(Path(sbom_path)),
    }
    manifest["signature"] = signing_key.sign(_canonical(manifest)).hex()
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def verify_release(manifest_path: str | Path, *, public_key: Ed25519PublicKey) -> None:
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    if document.get("format") != "xurcroh-release-v2":
        raise TransparencyError("unsupported release-manifest format")
    signature = document.pop("signature", None)
    if not isinstance(signature, str):
        raise TransparencyError("release signature is missing")
    try:
        public_key.verify(bytes.fromhex(signature), _canonical(document))
    except (InvalidSignature, ValueError) as error:
        raise TransparencyError("release signature is invalid") from error
    sbom_path = document.get("sbom_path")
    sbom_digest = document.get("sbom_sha256")
    if not isinstance(sbom_path, str) or not isinstance(sbom_digest, str):
        raise TransparencyError("release SBOM identity is missing")
    sbom = Path(sbom_path)
    if not sbom.is_file() or _digest(sbom) != sbom_digest:
        raise TransparencyError("release SBOM digest mismatch")
    root = path.parent.resolve()
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise TransparencyError("release artifact list is invalid")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise TransparencyError("release artifact entry is invalid")
        relative = Path(artifact.get("path", ""))
        digest = artifact.get("sha256")
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(digest, str)
        ):
            raise TransparencyError("release artifact path is outside the release directory")
        artifact_path = (root / relative).resolve()
        if not artifact_path.is_relative_to(root):
            raise TransparencyError("release artifact path is outside the release directory")
        if not artifact_path.is_file() or _digest(artifact_path) != digest:
            raise TransparencyError(f"release artifact digest mismatch: {relative}")
