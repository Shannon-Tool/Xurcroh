from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xurcroh.protocol import FORMAT_VERSION, PROTOCOL_LABEL, ProtocolVersion


def test_shared_protocol_version_is_explicitly_versioned() -> None:
    assert FORMAT_VERSION == 1
    assert PROTOCOL_LABEL == "xurcroh-v1"
    assert ProtocolVersion.parse("xurcroh-v1").value == FORMAT_VERSION


def test_unknown_protocol_version_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported protocol"):
        ProtocolVersion.parse("xurcroh-v2")


def test_requirement_ids_are_unique_and_complete() -> None:
    matrix_path = Path(__file__).parents[1] / "docs/requirements/matrix.yaml"
    document = yaml.safe_load(matrix_path.read_text())
    requirements = document["requirements"]
    ids = [entry["id"] for entry in requirements]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 90
    assert all(entry["status"] in document["status_legend"] for entry in requirements)
    assert all(entry["source"] and entry["requirement"] for entry in requirements)
