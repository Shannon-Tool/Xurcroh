from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from xurcroh.config import WorkerConfig
from xurcroh.fortress.deployment import AsyncSSHExecutor


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_production_ssh_executor_executes_and_uploads_over_real_sshd(tmp_path: Path) -> None:
    sshd = shutil.which("sshd")
    ssh_keygen = shutil.which("ssh-keygen")
    ssh_keyscan = shutil.which("ssh-keyscan")
    if sshd is None or ssh_keygen is None or ssh_keyscan is None:
        pytest.skip("loopback OpenSSH server tools are unavailable")
    host_key = tmp_path / "host_ed25519"
    client_key = tmp_path / "client_ed25519"
    for path in (host_key, client_key):
        subprocess.run(  # noqa: S603
            [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(path)], check=True
        )
    authorized = tmp_path / "authorized_keys"
    authorized.write_text(client_key.with_suffix(".pub").read_text())
    port = _free_port()
    pid_file = tmp_path / "sshd.pid"
    config = tmp_path / "sshd_config"
    config.write_text(
        f"ListenAddress 127.0.0.1\nPort {port}\nHostKey {host_key}\n"
        f"AuthorizedKeysFile {authorized}\nPasswordAuthentication no\n"
        f"PermitRootLogin yes\nUsePAM no\nStrictModes no\nPidFile {pid_file}\n"
        "Subsystem sftp internal-sftp\n"
    )
    log_path = tmp_path / "sshd.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(  # noqa: S603
            [sshd, "-D", "-e", "-f", str(config)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 10
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"loopback sshd exited: {log_path.read_text()}")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("loopback sshd did not start")
            time.sleep(0.05)
        known_hosts = tmp_path / "known_hosts"
        scan = subprocess.run(  # noqa: S603
            [ssh_keyscan, "-p", str(port), "127.0.0.1"],
            capture_output=True,
            text=True,
            check=True,
        )
        known_hosts.write_text(scan.stdout)
        worker = WorkerConfig(
            name="loopback",
            host="127.0.0.1",
            ssh={
                "username": "root",
                "port": port,
                "key_path": str(client_key),
                "known_hosts_path": str(known_hosts),
            },
        )
        executor = AsyncSSHExecutor()
        remote = tmp_path / "remote"
        executor.run(worker, f"mkdir -p {remote}")
        source = tmp_path / "artifact.bin"
        source.write_bytes(b"real-ssh-artifact")
        executor.upload(worker, [source], str(remote / "artifact.bin"))
        assert executor.run(worker, f"cat {remote}/artifact.bin") == "real-ssh-artifact"
    finally:
        process.terminate()
        process.wait(timeout=10)
