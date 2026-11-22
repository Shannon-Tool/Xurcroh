#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.transport import (
    MutualTLSIdentity,
    RoleEndpoint,
    _RoleTransport,
    encode_role_artifact,
    native_compiled_weight_share,
)
from xurcroh.kem import HybridKemKey


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)  # noqa: S603


def smoke(root: Path, image: str, skip_build: bool) -> dict[str, object]:
    if not skip_build:
        run(
            [
                "docker",
                "build",
                "--file",
                str(root / "deploy/fortress-worker.Dockerfile"),
                "--tag",
                image,
                str(root),
            ]
        )
    name = f"xurcroh-fortress-smoke-{uuid.uuid4().hex[:12]}"
    port = free_udp_port()
    with tempfile.TemporaryDirectory(prefix="xurcroh-container-smoke-") as directory:
        artifacts = Path(directory)
        (artifacts / "tls").mkdir()
        scratch = artifacts / "scratch"
        scratch.mkdir()
        private = HybridKemKey.generate()
        public = private.public_key
        (artifacts / "recipient.key").write_bytes(private.to_bytes())
        operation = compile_operation(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            threshold=1,
            slice_count=1,
            share_count=2,
        )
        (artifacts / "projection.json").write_bytes(encode_role_artifact(operation, 0, public))
        run(
            [
                sys.executable,
                str(root / "scripts/gpu/generate_tls.py"),
                "--output",
                str(artifacts / "tls"),
                "--service=worker=localhost,localhost",
            ]
        )
        for path in artifacts.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                name,
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=10001:10001",
                "--mount",
                f"type=bind,source={scratch},target=/runtime-data/scratch",
                "--mount",
                f"type=bind,source={artifacts},target=/runtime-data/shards,readonly",
                "--publish",
                f"127.0.0.1:{port}:9444/udp",
                image,
                "--artifact-dir=/runtime-data/shards",
                "--recipient-key=/runtime-data/shards/recipient.key",
                "--certificate=/runtime-data/shards/tls/worker.crt",
                "--private-key=/runtime-data/shards/tls/worker.key",
                "--ca=/runtime-data/shards/tls/ca.crt",
                "--host=0.0.0.0",
                "--port=9444",
                "--device=cpu",
                "--ready-file=/runtime-data/scratch/ready",
                "--pid-file=/runtime-data/scratch/role.pid",
                "--exit-status-file=/runtime-data/scratch/role.exit",
            ]
        )
        try:
            deadline = time.monotonic() + 120
            while not (scratch / "ready").is_file():
                if time.monotonic() >= deadline:
                    logs = run(["docker", "logs", name])
                    raise TimeoutError(
                        f"container did not become ready: {logs.stdout}\n{logs.stderr}"
                    )
                time.sleep(0.2)
            transport = _RoleTransport(
                RoleEndpoint(0, 0, f"127.0.0.1:{port}", public, "container"),
                MutualTLSIdentity(
                    certificates=(
                        (artifacts / "tls/client.crt").read_bytes(),
                        (artifacts / "tls/ca.crt").read_bytes(),
                    ),
                    private_key=(artifacts / "tls/client.key").read_bytes(),
                    pinned_ca=(artifacts / "tls/ca.crt").read_bytes(),
                ),
                "projection",
                rotation_interval=64,
                frame_bytes=16_384,
            )
            activation = torch.tensor([5, 7], dtype=torch.int64)
            try:
                actual, _, _ = transport.execute(activation, "container-smoke", 0)
            finally:
                transport.close()
            expected = native_compiled_weight_share(operation.shares[0])
            import xurcroh_quic

            reference = xurcroh_quic.execute_fortress_role([5, 7], expected)
            if actual != reference:
                raise ValueError(f"container product mismatch: {actual} != {reference}")
            run(["docker", "stop", "--time=30", name])
            status = run(["docker", "wait", name]).stdout.strip()
            if status != "0":
                raise RuntimeError(f"container exited with status {status}")
            return {
                "image": image,
                "product": actual,
                "worker_exit_code": int(status),
                "container_removed": True,
            }
        finally:
            subprocess.run(  # noqa: S603
                ["docker", "rm", "--force", name],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and exercise the sealed Fortress worker image"
    )
    parser.add_argument("--image", default="xurcroh-fortress-worker:local")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = smoke(root, args.image, args.skip_build)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
