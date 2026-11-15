from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.gpu.deploy_two_node import options

from xurcroh.config import WorkerConfig

SSH = shutil.which("ssh")
if SSH is None:
    raise RuntimeError("ssh is required for datagram tunnel deployment")
SSH_PATH = str(SSH)


def free_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_detached(worker: dict[str, Any], command: str) -> None:
    subprocess.run(  # noqa: S603
        [
            SSH_PATH,
            "-f",
            *options(worker),
            f"{worker['ssh_user']}@{worker['host']}",
            command,
        ],
        check=True,
    )


def wait_remote_port(worker: dict[str, Any], path: Path) -> int:
    deadline = time.monotonic() + 60
    while True:
        result = subprocess.run(  # noqa: S603
            [
                SSH_PATH,
                *options(worker),
                f"{worker['ssh_user']}@{worker['host']}",
                f"cat {shlex.quote(str(path))}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
        if time.monotonic() >= deadline:
            raise TimeoutError(f"remote datagram relay did not start for {worker['name']}")
        time.sleep(0.2)


def wait_local_port(path: Path, process: subprocess.Popen[bytes]) -> int:
    deadline = time.monotonic() + 30
    while True:
        if path.is_file() and path.read_text().strip().isdigit():
            return int(path.read_text().strip())
        if process.poll() is not None:
            raise RuntimeError(f"local datagram relay exited with status {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError("local datagram relay did not start")
        time.sleep(0.05)


def wait_tcp_forward(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    while True:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        if process.poll() is not None:
            raise RuntimeError(f"SSH datagram forward exited with status {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError("SSH datagram forward did not start")
        time.sleep(0.05)


def start_datagram_tunnels(
    workers: list[dict[str, Any]],
    remote_root: str,
    sdk: dict[str, Any],
    output: Path,
    *,
    target_port: int = 9444,
    target_ports: list[int] | None = None,
    tunnel_name: str = "relay",
) -> None:
    ports = target_ports or [target_port] * len(workers)
    if (
        len(ports) != len(workers)
        or any(not 1 <= port <= 65535 for port in ports)
        or not tunnel_name
        or "/" in tunnel_name
    ):
        raise ValueError("datagram tunnel requires valid per-worker ports and a simple name")
    output.mkdir(parents=True, exist_ok=True)
    started: list[dict[str, Any]] = []
    try:
        for index, worker in enumerate(workers):
            root = shlex.quote(remote_root)
            address = Path(remote_root) / f"{tunnel_name}.address"
            pid_path = Path(remote_root) / f"{tunnel_name}.pid"
            python = shlex.quote(str(worker.get("python_path", ".venv-cuda/bin/python")))
            start_detached(
                worker,
                f"cd {root} && setsid -f env {python} -m "
                "xurcroh.fortress.datagram_tunnel server --listen-host 127.0.0.1 "
                f"--listen-port 0 --target-host 127.0.0.1 --target-port {ports[index]} "
                f"--address-file {shlex.quote(str(address))} "
                f"--pid-file {shlex.quote(str(pid_path))} </dev/null "
                f">{root}/{shlex.quote(tunnel_name)}.log 2>&1",
            )
            remote_port = wait_remote_port(worker, address)
            forward_port = free_tcp_port()
            log_path = output / f"{worker['name']}-ssh.log"
            with log_path.open("wb") as log:
                forward = subprocess.Popen(  # noqa: S603
                    [
                        SSH_PATH,
                        "-N",
                        "-L",
                        f"127.0.0.1:{forward_port}:127.0.0.1:{remote_port}",
                        *options(worker),
                        f"{worker['ssh_user']}@{worker['host']}",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            wait_tcp_forward(forward_port, forward)
            local_address = output / f"{worker['name']}.address"
            with (output / f"{worker['name']}.log").open("wb") as log:
                local = subprocess.Popen(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "xurcroh.fortress.datagram_tunnel",
                        "client",
                        "--listen-host",
                        "127.0.0.1",
                        "--listen-port",
                        "0",
                        "--target-host",
                        "127.0.0.1",
                        "--target-port",
                        str(forward_port),
                        "--address-file",
                        str(local_address),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            local_port = wait_local_port(local_address, local)
            worker["datagram_tunnel"] = {
                "ssh_forward_pid": forward.pid,
                "local_relay_pid": local.pid,
                "local_endpoint": f"127.0.0.1:{local_port}",
                "remote_pid_path": str(pid_path),
            }
            configured = sdk["cluster"][index]
            endpoint = f"quic://127.0.0.1:{local_port}"
            if isinstance(configured, WorkerConfig):
                configured.endpoint = endpoint
            else:
                configured["endpoint"] = endpoint
            started.append(worker)
    except Exception:
        close_datagram_tunnels(started)
        raise


def close_datagram_tunnels(workers: list[dict[str, Any]]) -> None:
    for worker in workers:
        tunnel = worker.get("datagram_tunnel")
        if not isinstance(tunnel, dict):
            continue
        for key in ("local_relay_pid", "ssh_forward_pid"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(tunnel[key]), signal.SIGTERM)
