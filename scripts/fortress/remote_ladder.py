#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
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


def remote_command(remote_root: str, revision: str) -> str:
    root = shlex.quote(remote_root)
    return (
        f"cd {root} && set -o pipefail && "
        f"env PYTHONUNBUFFERED=1 XURCROH_SOURCE_REVISION={revision} "
        f".venv-cuda/bin/python scripts/fortress/run_ladder.py {root}/template.json "
        f"--provisioned-config {root}/workers.json --state {root}/provider-state.json "
        f"--output {root}/run --signing-key {root}/evidence.key --uncapped "
        f"2>&1 | tee {root}/run.log"
    )


def source_revision() -> str:
    if subprocess.run(  # noqa: S603
        [GIT, "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout:
        raise RuntimeError("commit all changes before gateway deployment")
    return subprocess.run(  # noqa: S603
        [GIT, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def run(args: argparse.Namespace) -> Path:
    revision = source_revision()
    template = json.loads(args.template.read_text())
    checkpoint = Path(template["checkpoint_path"])
    if not checkpoint.is_dir():
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
    ssh(gateway, f"cd {root} && ./scripts/gpu/bootstrap.sh")
    scp(gateway, [checkpoint], root, recursive=True)
    remote_template = {**template, "checkpoint_path": f"{root}/{checkpoint.name}"}
    remote_workers = worker_configuration(provisioned, f"{root}/worker-ssh-key")
    staged = {
        "template.json": remote_template,
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
    scp(gateway, [worker_key, args.signing_key], root)
    worker_name = shlex.quote(worker_key.name)
    signing_name = shlex.quote(args.signing_key.name)
    ssh(
        gateway,
        f"if test {worker_name} != worker-ssh-key; then "
        f"mv {root}/{worker_name} {root}/worker-ssh-key; fi && "
        f"if test {signing_name} != evidence.key; then "
        f"mv {root}/{signing_name} {root}/evidence.key; fi && "
        f"chmod 600 {root}/worker-ssh-key {root}/evidence.key",
    )
    try:
        ssh(gateway, remote_command(root, revision))
    finally:
        for remote, local in [
            (f"{root}/run.log", "run.log"),
            (f"{root}/run/evidence.json", "evidence.json"),
            (f"{root}/run/deployment.json", "deployment.json"),
            (f"{root}/run/manifest.json", "verified-manifest.json"),
        ]:
            try:
                ssh(gateway, f"test -f {remote}")
                download(gateway, remote, output / local)
            except (OSError, subprocess.CalledProcessError):
                if local in {"evidence.json", "deployment.json", "verified-manifest.json"}:
                    raise
    evidence = output / "evidence.json"
    private = Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
    verify_rung_evidence(evidence, public_key=private.public_key())
    print(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Fortress ladder through a trusted gateway")
    parser.add_argument("--gateway-config", required=True, type=Path)
    parser.add_argument("--worker-config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--worker-ssh-key", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
