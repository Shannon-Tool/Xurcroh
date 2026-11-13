from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def load_module():
    path = Path(__file__).resolve().parents[2] / "scripts/fortress/deploy_ladder.py"
    spec = importlib.util.spec_from_file_location("deploy_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipient_private_key_must_match_manifest(tmp_path: Path) -> None:
    module = load_module()
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    path = tmp_path / "role.key"
    path.write_bytes(private.private_bytes_raw())
    assert module.recipient_private_key(path, public) == private.private_bytes_raw()
    path.write_bytes(X25519PrivateKey.generate().private_bytes_raw())
    with pytest.raises(ValueError, match="does not match"):
        module.recipient_private_key(path, public)


def test_remote_command_executes_real_fortress_worker() -> None:
    module = load_module()
    command = module.start_command("/srv/fortress", "role-0-0")
    assert ".venv-cuda/bin/python -m xurcroh.fortress.worker_cli" in command
    assert "--artifact-dir /srv/fortress/role-artifacts" in command
    assert "--host 0.0.0.0 --port 9444" in command
    assert "--recipient-key /srv/fortress/recipient.key" in command
    assert "--exit-status-file /srv/fortress/role.exit" in command


def test_per_worker_ssh_configuration_is_normalized() -> None:
    module = load_module()
    document = {
        "workers": [
            {
                "ssh": {
                    "username": "ubuntu",
                    "port": 2222,
                    "key_path": "/keys/worker",
                    "known_hosts_path": "/keys/known_hosts",
                }
            }
        ]
    }
    worker = type("Worker", (), {"name": "role", "endpoint": "192.0.2.1:9444"})()
    normalized = module.ssh_worker(document, worker, 0)
    assert normalized["host"] == "192.0.2.1"
    assert normalized["ssh_user"] == "ubuntu"
    assert normalized["ssh_port"] == 2222
    assert normalized["ssh_key"] == "/keys/worker"
    assert normalized["known_hosts"] == "/keys/known_hosts"


def test_worker_shutdown_collects_actual_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    commands = []

    class Result:
        stdout = "0\n"
        returncode = 0

    def run(command, **kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr(module.subprocess, "run", run)
    worker = {
        "name": "role",
        "host": "192.0.2.1",
        "ssh_user": "ubuntu",
        "ssh_port": 22,
    }
    assert module.stop_workers([worker], "a" * 40) == [0]
    assert any("role.exit" in argument for command in commands for argument in command)


def test_hardware_validation_requires_a_reported_gpu_per_worker() -> None:
    module = load_module()
    reports = [{"gpus": [{"name": "RTX 4090"}]} for _ in range(4)]
    assert module.hardware_validated(reports, 4) is True
    reports[2]["gpus"] = []
    with pytest.raises(ValueError, match="hardware"):
        module.hardware_validated(reports, 4)


def test_ssh_waits_for_transport_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    results = iter(
        [
            type("Result", (), {"returncode": 255, "stderr": "Connection refused"})(),
            type("Result", (), {"returncode": 0, "stderr": ""})(),
        ]
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    module.wait_for_ssh({"ssh_user": "root", "host": "192.0.2.1"}, timeout=60)


def test_ssh_rejects_authentication_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    result = type("Result", (), {"returncode": 255, "stderr": "Permission denied"})()
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="authentication"):
        module.wait_for_ssh({"ssh_user": "root", "host": "192.0.2.1"}, timeout=60)


def test_host_key_is_pinned_per_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    result = type(
        "Result",
        (),
        {"returncode": 0, "stdout": "[192.0.2.1]:22 ssh-ed25519 public-key\n", "stderr": ""},
    )()
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: result)
    worker = {"host": "192.0.2.1", "ssh_port": 22}
    module.pin_host_key(worker, tmp_path / "known_hosts")
    assert worker["known_hosts"] == str(tmp_path / "known_hosts")
    assert "public-key" in (tmp_path / "known_hosts").read_text()
