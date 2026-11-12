#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: S607, S108
import argparse
import json
import shlex
import subprocess
from pathlib import Path

import deploy_two_node as deploy
from generate_tls import generate

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-prefix", default="/tmp/xurcroh-fp32-stage")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    workers = config["workers"]
    if len(workers) != 1:
        raise ValueError("FP32 sequential validation requires exactly one worker")
    worker = workers[0]
    manifest = json.loads((args.compiled / "manifest.json").read_text())
    stage_count = int(manifest["stage_count"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != config.get("revision", revision):
        raise RuntimeError("deployment config revision does not match HEAD")
    args.output.mkdir(parents=True, exist_ok=True)
    generate(args.output / "tls", {worker["name"]: [worker["public_host"]]})
    archive = subprocess.run(
        ["git", "archive", "--format=tar.gz", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout

    for index in range(stage_count):
        root = f"{args.remote_prefix}-{index}"
        quoted_root = shlex.quote(root)
        deploy.ssh(
            worker,
            f"rm -rf {quoted_root} && mkdir -p {quoted_root}/worker-artifacts {quoted_root}/tls",
        )
        deploy.ssh(worker, f"tar -xzf - -C {quoted_root}", input_data=archive)
        if index == 0:
            if not args.skip_bootstrap:
                deploy.ssh(worker, f"cd {quoted_root} && ./scripts/gpu/bootstrap.sh")
            if args.wheel is not None:
                deploy.scp(worker, [args.wheel], f"{root}/")
                deploy.ssh(
                    worker,
                    f"cd {quoted_root} && "
                    ".venv-cuda/bin/pip install --no-cache-dir --no-deps "
                    f"{shlex.quote(args.wheel.name)}",
                )
        else:
            deploy.ssh(
                worker,
                "cp -al "
                f"{shlex.quote(f'{args.remote_prefix}-0/.venv-cuda')} "
                f"{quoted_root}/.venv-cuda",
            )
        deploy.ssh(
            worker,
            f"cd {quoted_root} && "
            ".venv-cuda/bin/pip install --no-cache-dir --no-deps --no-build-isolation -e .",
        )
        deploy.scp(
            worker,
            [
                args.compiled / "manifest.json",
                args.compiled / f"stage-{index}.xurcroh",
                args.compiled / f"stage-{index}.key.json",
            ],
            f"{root}/worker-artifacts/",
        )
        deploy.scp(
            worker,
            [args.compiled / "identities" / f"stage-{index}.key"],
            f"{root}/stage.identity",
        )
        deploy.scp(worker, [args.compiled / "worker-checkpoint"], f"{root}/", recursive=True)
        name = worker["name"]
        deploy.scp(
            worker,
            [
                args.output / "tls" / "ca.crt",
                args.output / "tls" / f"{name}.crt",
                args.output / "tls" / f"{name}.key",
            ],
            f"{root}/tls/",
        )
        deploy.ssh(worker, f"chmod 600 {quoted_root}/stage.identity {quoted_root}/tls/{name}.key")
        print(f"prepared stage {index} at revision {revision}", flush=True)


if __name__ == "__main__":
    main()
