from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from xurcroh import release
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


def _native_wheel(path: Path, workspace: Path) -> bytes:
    sbom_name = "xurcroh_quic-0.1.0.dist-info/sboms/xurcroh-transport.cyclonedx.json"
    record_name = "xurcroh_quic-0.1.0.dist-info/RECORD"
    native_name = "xurcroh_quic/xurcroh_quic.cpython-312-x86_64-linux-gnu.so"
    sbom = {
        "bomFormat": "CycloneDX",
        "metadata": {
            "timestamp": "2023-11-14T22:13:20.000000000Z",
            "component": {
                "bom-ref": (f"path+file://{workspace}/crates/transport#xurcroh-transport@0.1.0")
            },
        },
        "components": [{"type": "library", "name": "quinn", "version": "0.11.12"}],
    }
    with ZipFile(path, "w") as wheel:
        for name, payload in (
            (native_name, b"identical-native-payload"),
            (sbom_name, json.dumps(sbom, indent=2, sort_keys=True).encode()),
            (record_name, f"{record_name},,\n".encode()),
        ):
            info = ZipInfo(name, date_time=(2023, 11, 14, 22, 13, 20))
            info.compress_type = 8
            wheel.writestr(info, payload)
    return native_name.encode()


def test_wheel_sbom_workspace_paths_are_reproducible(tmp_path: Path) -> None:
    wheels = []
    for name in ("first", "second"):
        workspace = tmp_path / name / "checkout"
        workspace.mkdir(parents=True)
        wheel = tmp_path / f"{name}.whl"
        native_name = _native_wheel(wheel, workspace).decode()
        release.normalize_wheel_sbom(wheel, project_root=workspace)
        wheels.append(wheel)

    assert wheels[0].read_bytes() == wheels[1].read_bytes()
    with ZipFile(wheels[0]) as wheel:
        assert all(info.compress_type == 0 for info in wheel.infolist())
        sbom_name = next(name for name in wheel.namelist() if name.endswith(".cyclonedx.json"))
        sbom = json.loads(wheel.read(sbom_name))
        assert sbom["metadata"]["component"]["bom-ref"].startswith(
            "path+file:///workspace/crates/transport#"
        )
        assert wheel.read(native_name) == b"identical-native-payload"
        record_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/RECORD"))
        records = list(csv.reader(wheel.read(record_name).decode().splitlines()))
        for name, digest, size in records:
            if name == record_name:
                assert digest == size == ""
                continue
            payload = wheel.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
            assert digest == f"sha256={expected.decode()}"
            assert int(size) == len(payload)


def test_native_wheel_build_normalizes_sbom_paths() -> None:
    script = Path("scripts/build-wheels.sh").read_text()
    assert "scripts/normalize-wheel-sbom.py" in script
    assert '"$OUT"/*.whl' in script
