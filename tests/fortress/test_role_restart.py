"""Stopping a managed role must cope with a role that is already dead or will not die."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from xurcroh.errors import ClusterUnavailableError
from xurcroh.fortress.deployment import FortressDeployment


class _LocalExecutor:
    def run(self, worker: object, command: str, timeout: float = 30) -> str:
        return subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        ).stdout


def _deployment(tmp_path: Path) -> tuple[FortressDeployment, SimpleNamespace, Path]:
    deployment = FortressDeployment.__new__(FortressDeployment)
    deployment._executor = _LocalExecutor()  # type: ignore[assignment]
    deployment._remote_root = str(tmp_path)
    role = SimpleNamespace(
        model_index=0, worker=SimpleNamespace(name="cpu-0"), service_name="model-0-cpu-0"
    )
    role_root = tmp_path / "model-0" / "cpu-0"
    role_root.mkdir(parents=True)
    return deployment, role, role_root


def test_a_killed_role_that_was_never_reaped_counts_as_stopped(tmp_path: Path) -> None:
    """A role killed with SIGKILL writes no exit file. Where nothing reaps it (a container
    without an init process) it stays a zombie, which still answers ``kill -0``."""

    deployment, role, role_root = _deployment(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        status = Path(f"/proc/{child.pid}/status")
        while "State:\tZ" not in status.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        os.kill(child.pid, 0)  # the zombie still answers a liveness signal
        (role_root / "role.pid").write_text(str(child.pid))

        started = time.monotonic()
        deployment._stop_role(role, timeout=3)  # type: ignore[arg-type]
        assert time.monotonic() - started < 2
    finally:
        child.wait()


def test_restart_proceeds_after_force_killing_a_role_that_ignores_termination(
    tmp_path: Path,
) -> None:
    deployment, role, role_root = _deployment(tmp_path)
    stubborn = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(120)"
    )

    def start() -> subprocess.Popen[str]:
        child = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", stubborn], stdout=subprocess.PIPE, text=True
        )
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        (role_root / "role.pid").write_text(str(child.pid))
        return child

    child = start()
    try:
        # Teardown reports a role that had to be killed.
        with pytest.raises(ClusterUnavailableError, match="forced termination"):
            deployment._stop_role(role, timeout=0.5)  # type: ignore[arg-type]
        assert child.wait(timeout=10) != 0
    finally:
        child.kill()
        child.wait()

    child = start()
    try:
        # A restart only needs the old process gone, however that was achieved.
        deployment._stop_role(role, timeout=0.5, force=True)  # type: ignore[arg-type]
        assert child.wait(timeout=10) != 0
    finally:
        child.kill()
        child.wait()
