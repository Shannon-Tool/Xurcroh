from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.fortress.worker_cli import (
    artifact_file_name,
    load_role_artifacts,
    main,
    record_exit_status,
)


def test_worker_cli_artifacts_preserve_operation_names(tmp_path: Path) -> None:
    operation = "model.layers.0.mlp.experts.0/gate_proj"
    (tmp_path / f"{artifact_file_name(operation)}.json").write_bytes(b"role-artifact")
    artifacts = load_role_artifacts(tmp_path)
    assert artifacts == {operation: b"role-artifact"}


def test_worker_cli_rejects_empty_artifact_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        load_role_artifacts(tmp_path)


def test_worker_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--recipient-key" in capsys.readouterr().out


def test_worker_exit_status_is_recorded_atomically(tmp_path: Path) -> None:
    path = tmp_path / "role.exit"
    record_exit_status(path, 0)
    assert path.read_text() == "0\n"
