#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def options(worker: dict[str, object]) -> list[str]:
    result = ["-p", str(worker.get("ssh_port", 22))]
    if worker.get("ssh_key"):
        result.extend(["-o", "IdentitiesOnly=yes"])
    if worker.get("known_hosts"):
        result.extend(
            [
                "-o",
                f"UserKnownHostsFile={Path(str(worker['known_hosts'])).expanduser()}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        )
    if worker.get("ssh_key"):
        result.extend(["-i", str(Path(str(worker["ssh_key"])).expanduser())])
    return result


def destination(worker: dict[str, object]) -> str:
    return f"{worker['ssh_user']}@{worker['host']}"


def ssh(worker: dict[str, object], command: str, *, input_data: bytes | None = None) -> None:
    subprocess.run(
        ["ssh", *options(worker), destination(worker), command],
        input=input_data,
        check=True,
    )


def scp(
    worker: dict[str, object], sources: list[Path], remote: str, *, recursive: bool = False
) -> None:
    scp_options = ["-P", str(worker.get("ssh_port", 22))]
    if worker.get("ssh_key"):
        scp_options.extend(["-o", "IdentitiesOnly=yes"])
    if worker.get("known_hosts"):
        scp_options.extend(
            [
                "-o",
                f"UserKnownHostsFile={Path(str(worker['known_hosts'])).expanduser()}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        )
    if worker.get("ssh_key"):
        scp_options.extend(["-i", str(Path(str(worker["ssh_key"])).expanduser())])
    subprocess.run(
        [
            "scp",
            *scp_options,
            *(["-r"] if recursive else []),
            *(str(path) for path in sources),
            f"{destination(worker)}:{remote}",
        ],
        check=True,
    )


def download(worker: dict[str, object], source: str, local_path: Path) -> None:
    scp_options = ["-P", str(worker.get("ssh_port", 22))]
    if worker.get("ssh_key"):
        scp_options.extend(["-o", "IdentitiesOnly=yes"])
    if worker.get("known_hosts"):
        scp_options.extend(
            [
                "-o",
                f"UserKnownHostsFile={Path(str(worker['known_hosts'])).expanduser()}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        )
    if worker.get("ssh_key"):
        scp_options.extend(["-i", str(Path(str(worker["ssh_key"])).expanduser())])
    subprocess.run(  # noqa: S603, S607
        ["scp", *scp_options, f"{destination(worker)}:{source}", str(local_path)],
        check=True,
    )


def prepare(index: int, worker: dict[str, object], config: dict[str, object], revision: str) -> str:
    remote_root = f"/tmp/xurcroh-{revision[:12]}"
    ssh(worker, f"rm -rf {shlex.quote(remote_root)} && mkdir -p {shlex.quote(remote_root)}")
    archive = subprocess.run(
        ["git", "archive", "--format=tar.gz", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    ssh(worker, f"tar -xzf - -C {shlex.quote(remote_root)}", input_data=archive)
    if os.environ.get("XURCROH_SKIP_BOOTSTRAP") != "1":
        ssh(worker, f"cd {shlex.quote(remote_root)} && ./scripts/gpu/bootstrap.sh")
    compiled = Path(str(config["compiled"]))
    remote_artifacts = f"{remote_root}/worker-artifacts"
    ssh(worker, f"mkdir -p {remote_artifacts} {remote_root}/tls")
    scp(
        worker,
        [
            compiled / "manifest.json",
            compiled / f"stage-{index}.xurcroh",
            compiled / f"stage-{index}.key.json",
        ],
        f"{remote_artifacts}/",
    )
    scp(worker, [compiled / "identities" / f"stage-{index}.key"], f"{remote_root}/stage.identity")
    scp(worker, [compiled / "worker-checkpoint"], f"{remote_root}/", recursive=True)
    tls = Path(str(config["output"])) / "tls"
    name = str(worker["name"])
    scp(worker, [tls / "ca.crt", tls / f"{name}.crt", tls / f"{name}.key"], f"{remote_root}/tls/")
    ssh(worker, f"chmod 600 {remote_root}/stage.identity {remote_root}/tls/{name}.key")
    return remote_root


def start(index: int, worker: dict[str, object], remote_root: str) -> None:
    name = str(worker["name"])
    environment = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29615",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LD_LIBRARY_PATH": (
            "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        ),
    }
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
    command = (
        f"cd {shlex.quote(remote_root)} && "
        f"nohup env {exports} .venv-cuda/bin/python -m xurcroh.stage_worker "
        f"--manifest {remote_root}/worker-artifacts/manifest.json --stage-index {index} "
        f"--stage-private-key {remote_root}/stage.identity --engine vllm "
        f"--checkpoint {remote_root}/worker-checkpoint --host 0.0.0.0 --port 9444 "
        f"--tls-cert {remote_root}/tls/{name}.crt --tls-key {remote_root}/tls/{name}.key "
        f"</dev/null >{remote_root}/stage.log 2>&1 & echo $! > {remote_root}/stage.pid"
    )
    ssh(worker, command)


def deploy(config_path: Path) -> Path:
    config = json.loads(config_path.read_text())
    workers = config["workers"]
    if len(workers) != 2 or len({worker["name"] for worker in workers}) != 2:
        raise ValueError("exactly two distinct workers are required")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout:
        raise RuntimeError(
            "commit all changes before deployment; git archive must match the tested SHA"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    output = Path(str(config["output"]))
    output.mkdir(parents=True, exist_ok=True)
    services = [
        f"--service={worker['name']}={worker['public_host']},{worker['name']}" for worker in workers
    ]
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/gpu/generate_tls.py"),
            "--output",
            str(output / "tls"),
            *services,
        ],
        check=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            roots = list(
                executor.map(
                    lambda item: prepare(item[0], item[1], config, revision),
                    enumerate(workers),
                )
            )
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda item: start(item[0], item[1], item[2]), zip(workers, roots)))
    except Exception:
        for worker in workers:
            root = f"/tmp/xurcroh-{revision[:12]}"
            subprocess.run(
                [
                    "ssh",
                    *options(worker),
                    destination(worker),
                    f"if test -f {root}/stage.pid; then kill $(cat {root}/stage.pid); fi",
                ],
                check=False,
            )
        raise
    deployment = {
        "schema_version": 1,
        "revision": revision,
        "model": config["model"],
        "checkpoint": config["checkpoint"],
        "manifest": str(Path(str(config["compiled"])) / "manifest.json"),
        "boundary_key_file": str(Path(str(config["compiled"])) / "client/boundary-key.hex"),
        "ca_file": str(output / "tls/ca.crt"),
        "workers": [
            {
                "name": worker["name"],
                "endpoint": f"https://{worker['public_host']}:{worker['tls_port']}",
                "remote_root": root,
            }
            for worker, root in zip(workers, roots)
        ],
    }
    path = output / "deployment.json"
    path.write_text(json.dumps(deployment, indent=2) + "\n")
    print(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    path = deploy(args.config)
    subprocess.run(
        ["python3", str(ROOT / "scripts/gpu/verify_two_node.py"), str(path), "--wait-only"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
