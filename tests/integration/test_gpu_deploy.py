from __future__ import annotations

import importlib.util
from pathlib import Path


def load_deploy_module():
    path = Path(__file__).resolve().parents[2] / "scripts/gpu/deploy_two_node.py"
    spec = importlib.util.spec_from_file_location("gpu_deploy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ssh_options_pin_identity_and_known_hosts(tmp_path: Path) -> None:
    module = load_deploy_module()
    key = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    options = module.options(
        {
            "ssh_port": 2222,
            "ssh_key": str(key),
            "known_hosts": str(known_hosts),
        }
    )
    assert options == [
        "-p",
        "2222",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        str(key),
    ]


def test_start_detaches_remote_worker_stdin(monkeypatch, tmp_path: Path) -> None:
    module = load_deploy_module()
    commands = []
    monkeypatch.setattr(module, "ssh", lambda worker, command: commands.append(command))
    worker = {
        "name": "worker-a",
        "rendezvous_host": "192.0.2.1",
    }
    module.start(0, worker, str(tmp_path / "worker"))
    assert len(commands) == 1
    assert "</dev/null" in commands[0]
