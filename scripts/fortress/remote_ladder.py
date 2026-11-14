#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.fortress.deploy_ladder import pin_host_key, wait_for_ssh
from scripts.gpu.deploy_two_node import download, scp, ssh

from xurcroh.fortress.ladder import verify_rung_evidence
from xurcroh.fortress.ladder_provider import destroy_ladder_instances
from xurcroh.providers.vast import StateStore

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("git is required for gateway deployment")


def worker_configuration(document: dict[str, Any], key_path: str) -> dict[str, Any]:
    result = copy.deepcopy(document)
    workers = result.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("worker configuration must contain workers")
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("worker configuration entries must be objects")
        worker["ssh_key"] = key_path
    return result


def remote_command(
    remote_root: str, revision: str, download_checkpoint: bool = False, smoke: bool = False
) -> str:
    root = shlex.quote(remote_root)
    prepare = (
        f".venv-cuda/bin/python scripts/fortress/prepare_checkpoint.py "
        f"--template {root}/template-request.json --checkpoint {root}/checkpoint "
        f"--output {root}/template.json --allow-download && "
        if download_checkpoint
        else ""
    )
    execute = (
        f".venv-cuda/bin/python scripts/fortress/grid_smoke.py "
        f"--template {root}/template.json --provisioned-config {root}/workers.json "
        f"--state {root}/provider-state.json --output {root}/run --uncapped "
        f"--defer-provider-teardown"
        if smoke
        else f".venv-cuda/bin/python scripts/fortress/run_ladder.py {root}/template.json "
        f"--provisioned-config {root}/workers.json --state {root}/provider-state.json "
        f"--output {root}/run --signing-key {root}/evidence.key --uncapped"
    )
    return (
        f"cd {root} && set -o pipefail && {{ {prepare}"
        f"env PYTHONUNBUFFERED=1 XURCROH_SOURCE_REVISION={revision} "
        f"{execute}; }} 2>&1 | tee {root}/run.log"
    )


def source_revision() -> str:
    if subprocess.run(  # noqa: S603
        [GIT, "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout:
        raise RuntimeError("commit all changes before gateway deployment")
    return subprocess.run(  # noqa: S603
        [GIT, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run(args: argparse.Namespace) -> Path:
    if not args.smoke and args.signing_key is None:
        raise ValueError("signed ladder evidence requires --signing-key")
    revision = source_revision()
    template = json.loads(args.template.read_text())
    checkpoint = Path(template["checkpoint_path"])
    if not args.download_checkpoint and not checkpoint.is_dir():
        raise ValueError("template checkpoint_path must be a trusted local directory")
    provisioned = json.loads(args.worker_config.read_text())
    gateway_workers = json.loads(args.gateway_config.read_text()).get("workers")
    if not isinstance(gateway_workers, list) or len(gateway_workers) != 1:
        raise ValueError("gateway configuration must contain exactly one worker")
    gateway = gateway_workers[0]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    root = f"/tmp/xurcroh-gateway-{revision[:12]}"  # noqa: S108
    pin_host_key(gateway, output / "gateway.known_hosts")
    wait_for_ssh(gateway)
    archive = subprocess.run(  # noqa: S603
        [GIT, "archive", "--format=tar.gz", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    ssh(gateway, f"rm -rf {root} && mkdir -p {root}")
    ssh(gateway, f"tar -xzf - -C {root}", input_data=archive)
    ssh(
        gateway,
        f"cd {root} && ./scripts/gpu/bootstrap.sh && "
        ".venv-cuda/bin/pip install --retries 10 --timeout 120 --no-cache-dir "
        "transformers==5.22.2 huggingface-hub==2.4.0 tokenizers==0.23.3 "
        "httpx2==2.16.0 cryptography==50.0.2 pillow==13.0.0 && "
        ".venv-cuda/bin/python -c "
        "'from transformers import AutoModelForMultimodalLM'",
    )
    if not args.download_checkpoint:
        scp(gateway, [checkpoint], root, recursive=True)
    checkpoint_name = "checkpoint" if args.download_checkpoint else checkpoint.name
    remote_template = {**template, "checkpoint_path": f"{root}/{checkpoint_name}"}
    remote_workers = worker_configuration(provisioned, f"{root}/worker-ssh-key")
    staged = {
        "template-request.json" if args.download_checkpoint else "template.json": remote_template,
        "workers.json": remote_workers,
        "provider-state.json": json.loads(args.state.read_text()),
    }
    for name, document in staged.items():
        (output / name).write_text(json.dumps(document, indent=2) + "\n")
    scp(
        gateway,
        [output / name for name in staged],
        root,
    )
    worker_key = args.worker_ssh_key or Path(gateway["ssh_key"])
    key_files = [worker_key] if args.smoke else [worker_key, args.signing_key]
    scp(gateway, key_files, root)
    worker_name = shlex.quote(worker_key.name)
    commands = [
        f"if test {worker_name} != worker-ssh-key; then "
        f"mv {root}/{worker_name} {root}/worker-ssh-key; fi",
        f"chmod 600 {root}/worker-ssh-key",
    ]
    if not args.smoke:
        signing_name = shlex.quote(args.signing_key.name)
        commands.extend(
            [
                f"if test {signing_name} != evidence.key; then "
                f"mv {root}/{signing_name} {root}/evidence.key; fi",
                f"chmod 600 {root}/evidence.key",
            ]
        )
    ssh(gateway, " && ".join(commands))
    downloads = (
        [
            (f"{root}/run.log", "run.log"),
            (f"{root}/run/run/smoke-report.json", "smoke-report.json"),
        ]
        if args.smoke
        else [
            (f"{root}/run.log", "run.log"),
            (f"{root}/run/evidence.json", "evidence.json"),
            (f"{root}/run/deployment.json", "deployment.json"),
            (f"{root}/run/manifest.json", "verified-manifest.json"),
        ]
    )
    try:
        ssh(
            gateway,
            remote_command(
                root,
                revision,
                download_checkpoint=args.download_checkpoint,
                smoke=args.smoke,
            ),
        )
    finally:
        for remote, local in downloads:
            try:
                ssh(gateway, f"test -f {remote}")
                download(gateway, remote, output / local)
            except (OSError, subprocess.CalledProcessError):
                if local != "run.log":
                    raise
    evidence = output / ("smoke-report.json" if args.smoke else "evidence.json")
    if args.smoke:
        report = json.loads(evidence.read_text())
        if "error_type" in report:
            raise RuntimeError(f"remote grid smoke failed: {report['error_type']}")
    else:
        private = Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
        verify_rung_evidence(evidence, public_key=private.public_key())
    print(evidence)
    return evidence


def run(args: argparse.Namespace) -> Path:
    try:
        return _run(args)
    finally:
        if args.smoke:
            store = StateStore(args.state)
            teardown = destroy_ladder_instances(store, uncapped=True)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "worker-teardown.json").write_text(json.dumps(teardown, indent=2) + "\n")
            watchdog_pid = store.snapshot().get("watchdog_pid")
            if watchdog_pid:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(watchdog_pid), signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Fortress ladder through a trusted gateway")
    parser.add_argument("--gateway-config", required=True, type=Path)
    parser.add_argument("--worker-config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--signing-key", type=Path)
    parser.add_argument("--worker-ssh-key", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
