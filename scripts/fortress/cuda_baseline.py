#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fortress.deploy_ladder import pin_host_key, source_revision, wait_for_ssh
from scripts.gpu.deploy_two_node import download, ssh

from xurcroh.fortress.ladder_provider import destroy_ladder_instances
from xurcroh.providers.vast import StateStore

GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("git is required for the CUDA baseline")


def run(args: argparse.Namespace) -> None:
    revision = source_revision()
    state_path = args.output / "state.json"
    config_path = args.output / "two-node.json"
    worker = None
    root = f"/tmp/xurcroh-forensics-{revision[:12]}"  # noqa: S108
    try:
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "scripts/vast/provision.py",
                str(state_path),
                *(["--uncapped"] if args.uncapped else []),
                "--count",
                "1",
                "--gpu-name",
                args.gpu_name,
                "--max-hourly-usd",
                str(args.max_hourly_usd),
                "--disk-gb",
                str(args.disk_gb),
                "--planned-hours",
                "1",
                "--boot-timeout",
                str(args.boot_timeout),
                "--ttl-seconds",
                str(args.ttl_seconds),
                "--image",
                args.image,
                "--ssh-key",
                str(args.ssh_key),
                "--output",
                str(args.output),
            ],
            check=True,
        )
        worker = json.loads(config_path.read_text())["workers"][0]
        pin_host_key(worker, args.output / "worker.known_hosts")
        wait_for_ssh(worker)
        archive = subprocess.run(  # noqa: S603
            [GIT, "archive", "--format=tar.gz", revision], check=True, capture_output=True
        ).stdout
        ssh(worker, f"rm -rf {root} && mkdir -p {root}")
        ssh(worker, f"tar -xzf - -C {root}", input_data=archive)
        ssh(worker, f"cd {root} && ./scripts/gpu/bootstrap.sh")
        ssh(
            worker,
            f"cd {root} && set -o pipefail && "
            f"PYTHON={root}/.venv-cuda/bin/python ./scripts/fortress/cuda_forensics.sh "
            f"2>&1 | tee {root}/artifacts/fortress-cuda-forensics.log",
        )
    finally:
        if worker is not None:
            for source, name in [
                (f"{root}/artifacts/hardware.json", "hardware.json"),
                (f"{root}/artifacts/fortress-cuda-forensics.log", "fortress-cuda-forensics.log"),
            ]:
                try:
                    download(worker, source, args.output / name)
                except (OSError, subprocess.CalledProcessError) as error:
                    print({"evidence_download": name, "error": type(error).__name__})
        if state_path.is_file():
            store = StateStore(state_path)
            print({"teardown": destroy_ladder_instances(store, uncapped=args.uncapped)})
            watchdog_pid = store.snapshot().get("watchdog_pid")
            if watchdog_pid:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(watchdog_pid), signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the physical Fortress CUDA baseline")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="vllm/vllm-openai:v0.15.0")
    parser.add_argument("--gpu-name", default="RTX 4090")
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--max-hourly-usd", type=float, default=2.0)
    parser.add_argument("--disk-gb", type=float, default=60)
    parser.add_argument("--boot-timeout", type=float, default=2400)
    parser.add_argument("--ttl-seconds", type=float, default=5400)
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
