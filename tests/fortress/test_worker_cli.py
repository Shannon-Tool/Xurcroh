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


def test_worker_close_uses_one_global_connection_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from xurcroh.fortress.worker import FortressRoleWorker

    clock = [0.0]
    joins: list[float] = []

    class Thread:
        def __init__(self, alive: bool) -> None:
            self.alive = alive

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            joins.append(timeout)
            clock[0] += timeout

    class Server:
        def close(self) -> None:
            return None

    class Sessions:
        def close(self) -> None:
            return None

    monkeypatch.setattr("xurcroh.fortress.worker.time.monotonic", lambda: clock[0])
    worker = FortressRoleWorker.__new__(FortressRoleWorker)
    worker._closed = threading.Event()
    worker.server = Server()
    worker._accept_thread = Thread(False)
    worker._threads = [Thread(True) for _ in range(20)]
    worker._lock = threading.Lock()
    worker._requests = {}
    worker._session_completed = {}
    worker._active_sessions = set()
    worker.sessions = Sessions()
    worker._cuda_weights = {}
    worker.device = "cpu"
    worker.artifacts = {}

    worker.close()

    assert len(joins) == 20
    assert clock[0] <= 10


def test_worker_exit_status_is_recorded_atomically(tmp_path: Path) -> None:
    path = tmp_path / "role.exit"
    record_exit_status(path, 0)
    assert path.read_text() == "0\n"
