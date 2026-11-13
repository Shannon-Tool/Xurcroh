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
