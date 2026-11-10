from __future__ import annotations

from xurcroh.config import WorkerConfig
from xurcroh.provisioning import build_worker_command


def test_worker_command_is_shell_quoted_and_contains_no_credentials() -> None:
    worker = WorkerConfig(
        name="worker-a",
        host="10.0.0.10",
        ssh={"username": "ubuntu", "key_path": "/private/key", "password": "secret"},
    )
    command = build_worker_command(
        worker,
        model_paths={"tiny": "/models/tiny"},
        controller="https://controller.internal:8443",
    )
    assert "secret" not in command
    assert "/private/key" not in command
    assert "worker-a" in command
    assert "/models/tiny" in command
