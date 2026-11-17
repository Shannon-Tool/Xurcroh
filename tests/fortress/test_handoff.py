from __future__ import annotations

from pathlib import Path

import pytest
from scripts.fortress.run_ladder import load_rung_manifest
from scripts.fortress.validate_handoff import validate


def test_all_public_ladder_templates_validate_offline() -> None:
    root = Path(__file__).parents[2]
    commands = [
        command
        for path in sorted((root / "configs/ladders").glob("*.yaml"))
        for command in validate(path)
    ]
    assert len(commands) == 21
    assert all("run_ladder.py" in command for command in commands)
    assert all("--model" in command for command in commands if "text-ladder" in command)


def test_multi_model_manifest_selects_exact_model() -> None:
    path = Path(__file__).parents[2] / "configs/ladders/text-ladder.yaml"
    selected = load_rung_manifest(path, "qwen-3.8-27b")
    assert selected["model"] == "qwen-3.8-27b"
    assert selected["rung"] == "L3"
    assert "rungs" not in selected


def test_multi_model_manifest_requires_explicit_model() -> None:
    path = Path(__file__).parents[2] / "configs/ladders/media-ladder.yaml"
    with pytest.raises(ValueError, match="--model"):
        load_rung_manifest(path, None)
