from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from xurcroh.config import SecurityConfig

PROBE = """
import ctypes, resource
from xurcroh.memory_host import protect_process
protect_process()
libc = ctypes.CDLL(None)
print(libc.prctl(3, 0, 0, 0, 0), *resource.getrlimit(resource.RLIMIT_CORE))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="process protection is Linux-specific")
def test_protect_process_disables_dumps_and_core_files() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True, timeout=60
    )

    assert result.stdout.split() == ["0", "0", "0"]


def test_role_worker_cli_hardens_the_process_before_reading_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.fortress import worker_cli

    order: list[str] = []
    monkeypatch.setattr(
        worker_cli, "protect_process", lambda *, lock_memory=False: order.append("protect")
    )

    def load(_directory: object) -> dict[str, bytes]:
        order.append("load")
        raise ValueError("stop")

    monkeypatch.setattr(worker_cli, "load_role_artifacts", load)
    arguments = ["--artifact-dir", str(tmp_path), "--recipient-key", str(tmp_path / "k")]
    arguments += ["--certificate", "c", "--private-key", "p", "--ca", "a"]

    with pytest.raises(ValueError, match="stop"):
        worker_cli.main(arguments)

    assert order == ["protect", "load"]


def test_process_hardening_is_on_by_default_and_recorded() -> None:
    assert SecurityConfig().process_hardening is True
    assert SecurityConfig(mode="development", process_hardening=False).process_hardening is False


def test_runtime_service_hardens_the_client_process(
    tiny_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Xurcroh import Xurcroh
    from xurcroh.runtime import service

    calls: list[bool] = []
    monkeypatch.setattr(service, "protect_process", lambda *, lock_memory=False: calls.append(True))

    with Xurcroh(cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}):
        pass
    assert calls == [True]

    calls.clear()
    cluster = {
        "endpoint": "local://cpu",
        "security": {"mode": "development", "process_hardening": False},
    }
    with Xurcroh(cluster=cluster, model_path={"tiny": tiny_model_path}):
        pass
    assert calls == []
