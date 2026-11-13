from __future__ import annotations

import json
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
