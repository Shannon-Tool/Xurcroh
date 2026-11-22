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
    worker._transcripts = {}
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


def test_worker_cli_requires_onion_wrapped_traffic_when_given_an_onion_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.fortress import worker_cli

    captured: dict[str, object] = {}

    def fake_worker(*_arguments: object, **options: object) -> None:
        captured.update(options)
        raise RuntimeError("stop")

    monkeypatch.setattr(worker_cli, "FortressRoleWorker", fake_worker)
    monkeypatch.setattr(worker_cli, "protect_process", lambda *, lock_memory=False: None)
    monkeypatch.setattr(worker_cli, "load_role_artifacts", lambda _directory: {"op": b"x"})
    for name in ("recipient", "certificate", "key", "ca"):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "onion").write_bytes(b"\x07" * 96)
    arguments = ["--artifact-dir", str(tmp_path), "--recipient-key", str(tmp_path / "recipient")]
    arguments += ["--certificate", str(tmp_path / "certificate"), "--ca", str(tmp_path / "ca")]
    arguments += ["--private-key", str(tmp_path / "key")]

    with pytest.raises(RuntimeError, match="stop"):
        worker_cli.main([*arguments, "--onion-key", str(tmp_path / "onion")])
    assert captured["onion_private_key"] == b"\x07" * 96

    captured.clear()
    with pytest.raises(RuntimeError, match="stop"):
        worker_cli.main(arguments)
    assert captured["onion_private_key"] is None

    (tmp_path / "onion").write_bytes(b"short")
    with pytest.raises(ValueError, match="96 bytes"):
        worker_cli.main([*arguments, "--onion-key", str(tmp_path / "onion")])


def test_worker_cli_passes_the_identity_key_to_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.fortress import worker_cli
    from xurcroh.signatures import HybridSigningKey

    captured: dict[str, object] = {}

    def fake_worker(*_arguments: object, **options: object) -> None:
        captured.update(options)
        raise RuntimeError("stop")

    monkeypatch.setattr(worker_cli, "FortressRoleWorker", fake_worker)
    monkeypatch.setattr(worker_cli, "protect_process", lambda *, lock_memory=False: None)
    monkeypatch.setattr(worker_cli, "load_role_artifacts", lambda _directory: {"op": b"x"})
    for name in ("recipient", "certificate", "key", "ca"):
        (tmp_path / name).write_bytes(name.encode())
    identity = HybridSigningKey.generate().to_bytes()
    (tmp_path / "identity").write_bytes(identity)
    arguments = ["--artifact-dir", str(tmp_path), "--recipient-key", str(tmp_path / "recipient")]
    arguments += ["--certificate", str(tmp_path / "certificate"), "--ca", str(tmp_path / "ca")]
    arguments += ["--private-key", str(tmp_path / "key")]

    with pytest.raises(RuntimeError, match="stop"):
        worker_cli.main([*arguments, "--identity-key", str(tmp_path / "identity")])
    assert captured["identity_key"] == identity

    captured.clear()
    with pytest.raises(RuntimeError, match="stop"):
        worker_cli.main(arguments)
    assert captured["identity_key"] is None
