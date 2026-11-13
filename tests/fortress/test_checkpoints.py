from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from xurcroh.fortress.checkpoints import checkpoint_metadata


def test_checkpoint_metadata_pins_revision_license_and_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "LICENSE").write_text("Apache-2.0\n")
    first = checkpoint_metadata(
        tmp_path,
        repository="org/model",
        revision="a" * 40,
        allow_download=False,
    )
    assert first["repository"] == "org/model"
    assert first["revision"] == "a" * 40
    assert first["license"] == "Apache-2.0"
    assert len(first["checkpoint_fingerprint"]) == 64
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    second = checkpoint_metadata(
        tmp_path,
        repository="org/model",
        revision="a" * 40,
        allow_download=False,
    )
    assert first["checkpoint_fingerprint"] != second["checkpoint_fingerprint"]


def test_prepare_checkpoint_cli_materializes_pinned_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "test"}))
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "LICENSE").write_text("Apache-2.0\n")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "repository": "org/model",
                "revision": "a" * 40,
                "license": "Apache-2.0",
                "checkpoint_path": "unused",
            }
        )
    )
    output = tmp_path / "template.json"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(root / "scripts/fortress/prepare_checkpoint.py"),
            "--template",
            str(request),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=True,
        cwd=root,
        capture_output=True,
    )
    document = json.loads(output.read_text())
    assert document["checkpoint_path"] == str(checkpoint.resolve())
    assert len(document["checkpoint_fingerprint"]) == 64


def test_checkpoint_metadata_rejects_floating_revision(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "LICENSE").write_text("Apache-2.0")
    with pytest.raises(ValueError, match="immutable"):
        checkpoint_metadata(
            tmp_path,
            repository="org/model",
            revision="main",
            allow_download=False,
        )
