#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from scripts.gpu.deploy_two_node import download, options, scp, ssh

from xurcroh.config import GPUConfig, ModelDefinition, WorkerConfig
from xurcroh.errors import ConfigurationError
from xurcroh.fortress.datagram_deploy import close_datagram_tunnels, start_datagram_tunnels
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import (
    place_rung_operations,
    sdk_configuration,
    write_operation_role_artifacts,
)
from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
from xurcroh.fortress.media_components import compile_media_checkpoint, discover_media_operations
from xurcroh.fortress.model_compiler import (
    compile_fortress_checkpoint,
    discover_fortress_operations,
    load_reap_saliency,
)
from xurcroh.kem import HybridKemKey, KemError
from xurcroh.preflight import preflight_checkpoint

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")
SSH = shutil.which("ssh")
if GIT is None or SSH is None:
    raise RuntimeError("git and ssh are required for Fortress deployment")
SSH_KEYSCAN = shutil.which("ssh-keyscan")
if SSH_KEYSCAN is None:
    raise RuntimeError("ssh-keyscan is required to pin worker host keys")


def recipient_private_key(path: Path, expected_public_hex: str) -> bytes:
    private = path.read_bytes()
    try:
        public = HybridKemKey.from_bytes(private).public_key.hex()
    except (KemError, ValueError) as error:
        raise ValueError(f"recipient key {path} is invalid") from error
    if public != expected_public_hex:
        raise ValueError(f"recipient key {path} does not match the manifest")
    return private


def endpoint_host(endpoint: str) -> str:
    parsed = urlsplit(f"quic://{endpoint}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"invalid Fortress endpoint: {endpoint}")
    return parsed.hostname


def validate_role_capacity(
    rung: LadderRung, checkpoint: Path, reports: list[dict[str, Any]]
) -> None:
    if len(reports) != len(rung.workers):
        raise ConfigurationError("role capacity preflight requires one hardware report per worker")
    workers = []
    for worker, report in zip(rung.workers, reports, strict=True):
        memories = [int(gpu["memory_bytes"]) for gpu in report.get("gpus", [])]
        if not memories:
            raise ConfigurationError(f"worker {worker.name} has no reported GPU capacity")
        workers.append(
            WorkerConfig(
                name=worker.name,
                endpoint=f"quic://{worker.endpoint}",
                gpu=GPUConfig(memory_gb=min(memories) / 1024**3),
            )
        )
    definition = ModelDefinition(
        path=str(checkpoint),
        engine="fortress",
        modality=rung.modality,
        fortress_shares=rung.share_count,
    )
    preflight = preflight_checkpoint(
        rung.model,
        definition,
        workers,
        require_known_memory=True,
        validate_roles=False,
    )
    if not preflight.supported:
        raise ConfigurationError("role capacity preflight failed: " + "; ".join(preflight.issues))


def hardware_validated(reports: list[dict[str, Any]], worker_count: int) -> bool:
    if len(reports) != worker_count or any(not report.get("gpus") for report in reports):
        raise ValueError("hardware validation requires one nonempty GPU report per worker")
    if any(
        str(report.get("cpu_vendor", "unknown")).lower() in {"", "unknown"} for report in reports
    ):
        raise ValueError("hardware validation requires a known CPU vendor per worker")
    return True


def start_command(remote_root: str, name: str, device: str = "cpu") -> str:
    root = shlex.quote(remote_root)
    return (
        f"cd {root} && setsid env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 "
        f".venv-cuda/bin/python -m xurcroh.fortress.worker_cli "
        f"--artifact-dir {root}/role-artifacts --recipient-key {root}/recipient.key "
        # A role signs session checkpoints when its identity key was uploaded.
        f"$(test -f {root}/identity.key && echo --identity-key {root}/identity.key) "
        f"--certificate {root}/tls/{shlex.quote(name)}.crt "
        f"--private-key {root}/tls/{shlex.quote(name)}.key --ca {root}/tls/ca.crt "
        f"--host 0.0.0.0 --port 9444 --device {shlex.quote(device)} "
        "--allow-unattested "
        f"--exit-status-file {root}/role.exit "
        f"</dev/null >{root}/role.log 2>&1 & "
        f"echo $! > {root}/role.pid"
    )


def start_worker(worker: dict[str, Any], remote_root: str, name: str) -> None:
    subprocess.run(  # noqa: S603
        [
            SSH,
            "-f",
            *options(worker),
            f"{worker['ssh_user']}@{worker['host']}",
            start_command(remote_root, name, device="cuda"),
        ],
        check=True,
    )


def ssh_worker(manifest: dict[str, Any], worker: Any, index: int) -> dict[str, Any]:
    raw_worker = manifest["workers"][index]
    settings = {
        **manifest.get("ssh", {}),
        **raw_worker.get("ssh", {}),
    }
    result = {
        "name": worker.name,
        "host": settings.get("host", endpoint_host(worker.endpoint)),
        "ssh_user": settings.get("username", manifest.get("ssh_user", "root")),
        "ssh_port": settings.get("port", manifest.get("ssh_port", 22)),
    }
    for source, destination in (("key_path", "ssh_key"), ("known_hosts_path", "known_hosts")):
        if source in settings:
            result[destination] = settings[source]
    return result


def source_revision() -> str:
    archived = os.environ.get("XURCROH_SOURCE_REVISION")
    if archived is not None:
        if len(archived) != 40 or any(
            character not in "0123456789abcdef" for character in archived
        ):
            raise ValueError("archived source revision must be 40 lowercase hexadecimal characters")
        return archived
    if subprocess.run(  # noqa: S603
        [GIT, "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout:
        raise RuntimeError("commit all changes before deployment; git archive must match HEAD")
    return subprocess.run(  # noqa: S603
        [GIT, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_archive() -> bytes:
    if os.environ.get("XURCROH_SOURCE_REVISION") is not None:
        archived = ROOT / "source.tar.gz"
        if not archived.is_file():
            raise RuntimeError("archived source deployment is missing source.tar.gz")
        return archived.read_bytes()
    revision = source_revision()
    return subprocess.run(  # noqa: S603
        [GIT, "archive", "--format=tar.gz", revision],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def pin_host_key(worker: dict[str, Any], path: Path, timeout: float = 900) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        deadline = time.monotonic() + timeout
        while True:
            result = subprocess.run(  # noqa: S603
                [
                    SSH_KEYSCAN,
                    "-T",
                    "10",
                    "-p",
                    str(worker.get("ssh_port", 22)),
                    worker["host"],
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            keys = "\n".join(
                line for line in result.stdout.splitlines() if line and not line.startswith("#")
            )
            if result.returncode == 0 and keys:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("could not obtain the worker SSH host key")
            time.sleep(5)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(keys + "\n")
    worker["known_hosts"] = str(path)


def wait_for_ssh(worker: dict[str, Any], timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run(  # noqa: S603
            [
                SSH,
                *options(worker),
                f"{worker['ssh_user']}@{worker['host']}",
                "true",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        if "permission denied" in result.stderr.lower():
            raise RuntimeError("SSH authentication failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("SSH transport did not become ready")
        time.sleep(5)


def prepare_worker(
    worker: Any,
    *,
    remote_root: str,
    package: Path,
    recipient_key: Path,
    tls: Path,
    host: str,
    archive: bytes,
    skip_bootstrap: bool,
) -> None:
    pin_host_key(worker, package.parent.parent / f"{worker['name']}.known_hosts")
    wait_for_ssh(worker)
    root = shlex.quote(remote_root)
    ssh(worker, f"rm -rf {root} && mkdir -p {root}/tls", input_data=None)
    ssh(worker, f"tar -xzf - -C {root}", input_data=archive)
    if not skip_bootstrap:
        ssh(worker, f"cd {root} && ./scripts/gpu/bootstrap.sh")
    scp(worker, [package], remote_root, recursive=True)
    ssh(worker, f"mv {root}/{shlex.quote(package.name)} {root}/role-artifacts")
    scp(worker, [recipient_key], f"{remote_root}/recipient.key")
    identity_key = recipient_key.with_name(f"{recipient_key.stem}.identity.key")
    if identity_key.is_file():
        scp(worker, [identity_key], f"{remote_root}/identity.key")
        ssh(worker, f"chmod 600 {root}/identity.key")
    scp(
        worker,
        [tls / "ca.crt", tls / f"{worker['name']}.crt", tls / f"{worker['name']}.key"],
        f"{remote_root}/tls/",
    )
    ssh(
        worker,
        f"chmod 600 {root}/recipient.key {root}/tls/{shlex.quote(worker['name'])}.key",
    )


def kill_workers(workers: list[Any], revision: str) -> None:
    for worker in workers:
        root = f"/tmp/xurcroh-fortress-{revision[:12]}"  # noqa: S108
        subprocess.run(  # noqa: S603
            [
                SSH,
                *options(worker),
                f"{worker['ssh_user']}@{worker['host']}",
                f"if test -f {root}/role.pid; then kill $(cat {root}/role.pid); fi; "
                f"for relay_pid in {root}/relay*.pid; do "
                'if test -f "$relay_pid"; then kill $(cat "$relay_pid"); fi; done',
            ],
            check=False,
        )
        close_datagram_tunnels([worker])


def stop_workers(workers: list[Any], revision: str, *, timeout: float = 60) -> list[int]:
    kill_workers(workers, revision)
    exit_codes = []
    for worker in workers:
        root = f"/tmp/xurcroh-fortress-{revision[:12]}"  # noqa: S108
        deadline = time.monotonic() + timeout
        while True:
            result = subprocess.run(  # noqa: S603
                [
                    SSH,
                    *options(worker),
                    f"{worker['ssh_user']}@{worker['host']}",
                    f"if test -f {root}/role.exit; then cat {root}/role.exit; fi",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                exit_codes.append(int(result.stdout.strip()))
                break
            if time.monotonic() >= deadline:
                subprocess.run(  # noqa: S603
                    [
                        SSH,
                        *options(worker),
                        f"{worker['ssh_user']}@{worker['host']}",
                        f"if test -f {root}/role.pid; then kill -KILL $(cat {root}/role.pid); fi",
                    ],
                    check=False,
                )
                exit_codes.append(124)
                break
            time.sleep(1)
    return exit_codes


def collect_role_resources(workers: list[Any], revision: str) -> dict[str, Any]:
    resources = {}
    root = shlex.quote(f"/tmp/xurcroh-fortress-{revision[:12]}")  # noqa: S108
    for worker in workers:
        command = (
            f"pid=$(cat {root}/role.pid 2>/dev/null || true); "
            'if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then '
            'ps -p "$pid" -o %cpu=,rss=; '
            "if command -v nvidia-smi >/dev/null 2>&1; then "
            "nvidia-smi --query-compute-apps=pid,used_memory "
            "--format=csv,noheader,nounits | "
            "awk -F', *' -v pid=\"$pid\" '$1 == pid {print $2}'; fi; fi"
        )
        result = subprocess.run(  # noqa: S603
            [
                SSH,
                *options(worker),
                f"{worker['ssh_user']}@{worker['host']}",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError(f"role {worker['name']} exited before resource collection")
        cpu, rss = lines[0].split()
        resources[worker["name"]] = {
            "running": True,
            "cpu_percent": float(cpu),
            "rss_bytes": int(rss) * 1024,
            "gpu_memory_bytes": sum(int(value.strip()) for value in lines[1:]) * 1024 * 1024,
        }
    return resources


def deploy(manifest_path: Path, output: Path, *, skip_bootstrap: bool = False) -> Path:
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("ladder manifest must be an object")
    rung = LadderRung.from_mapping(raw, allow_unverified_hardware=True)
    startup_concurrency = raw.get("startup_concurrency", 2)
    if (
        isinstance(startup_concurrency, bool)
        or not isinstance(startup_concurrency, int)
        or not 1 <= startup_concurrency <= 32
    ):
        raise ConfigurationError("startup_concurrency must be an integer from 1 to 32")
    revision = source_revision()
    checkpoint = Path(raw.get("checkpoint_path", ""))
    if not checkpoint.is_dir():
        raise ValueError("checkpoint_path must name a trusted local checkpoint directory")
    verification = rung.verification
    reap_keep_fraction = verification.get("reap_keep_fraction")
    reap_saliency = (
        load_reap_saliency(verification["reap_saliency_path"])
        if "reap_saliency_path" in verification
        else None
    )
    placement = None
    if rung.rung not in {"F0", "M0"} or any(worker.operation_roles for worker in rung.workers):
        if rung.modality in {"text", "vlm"}:
            operations = discover_fortress_operations(
                checkpoint,
                reap_keep_fraction=reap_keep_fraction,
                reap_saliency=reap_saliency,
            )
        else:
            adapter = MediaArchitectureAdapter.from_checkpoint(checkpoint, rung.modality)
            operations = discover_media_operations(str(checkpoint), adapter)
        rung, placement = place_rung_operations(rung, operations)
        raw["workers"] = [asdict(worker) for worker in rung.workers]
        raw["placement_epoch"] = rung.placement_epoch
        raw["max_provider_fraction"] = rung.max_provider_fraction
    key_directory = Path(
        raw.get("recipient_key_directory", manifest_path.parent / "recipient-keys")
    )
    for worker in rung.workers:
        recipient_private_key(key_directory / f"{worker.name}.key", worker.recipient_public)
    seed = (
        int(os.environ["XURCROH_FORTRESS_SEED"])
        if "XURCROH_FORTRESS_SEED" in os.environ
        else secrets.randbits(256)
    )
    output.mkdir(parents=True, exist_ok=True)
    client = output / "client"
    client.mkdir(exist_ok=True)
    seed_path = client / "fortress-seed"
    descriptor = os.open(seed_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{seed}\n")
    role_root = output / "roles"
    if rung.modality in {"text", "vlm"}:
        compile_fortress_checkpoint(
            checkpoint,
            seed=seed,
            share_count=rung.share_count,
            load_model=False,
            reap_keep_fraction=reap_keep_fraction,
            reap_saliency=reap_saliency,
            operation_sink=lambda name, operation: write_operation_role_artifacts(
                name, operation, rung, role_root
            ),
        )
    else:
        definition = ModelDefinition(
            path=str(checkpoint),
            engine="fortress",
            modality=rung.modality,
            fortress_seed=seed,
            fortress_shares=rung.share_count,
        )
        compile_media_checkpoint(
            definition,
            MediaArchitectureAdapter.from_checkpoint(checkpoint, rung.modality),
            operation_sink=lambda name, operation: write_operation_role_artifacts(
                name, operation, rung, role_root
            ),
        )
    packages = {worker.name: role_root / worker.name for worker in rung.workers}
    services = []
    for worker in rung.workers:
        host = endpoint_host(worker.endpoint)
        services.append(f"--service={worker.name}={host},{worker.name}")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/gpu/generate_tls.py"),
            "--output",
            str(output / "tls"),
            *services,
        ],
        check=True,
    )
    archive = source_archive()
    workers = [ssh_worker(raw, worker, index) for index, worker in enumerate(rung.workers)]
    remote_root = f"/tmp/xurcroh-fortress-{revision[:12]}"  # noqa: S108
    try:
        with ThreadPoolExecutor(max_workers=min(startup_concurrency, len(workers))) as executor:
            list(
                executor.map(
                    lambda item: prepare_worker(
                        item[1],
                        remote_root=remote_root,
                        package=packages[rung.workers[item[0]].name],
                        recipient_key=key_directory / f"{rung.workers[item[0]].name}.key",
                        tls=output / "tls",
                        host=endpoint_host(rung.workers[item[0]].endpoint),
                        archive=archive,
                        skip_bootstrap=skip_bootstrap,
                    ),
                    enumerate(workers),
                )
            )
        hardware_directory = output / "hardware"
        hardware_directory.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(startup_concurrency, len(workers))) as executor:
            list(
                executor.map(
                    lambda worker: download(
                        worker,
                        f"{remote_root}/artifacts/hardware.json",
                        hardware_directory / f"{worker['name']}.json",
                    ),
                    workers,
                )
            )
        hardware_reports = [
            json.loads((hardware_directory / f"{worker['name']}.json").read_text())
            for worker in workers
        ]
        hardware_validated(hardware_reports, len(workers))
        for raw_worker, report in zip(raw["workers"], hardware_reports, strict=True):
            raw_worker["cpu_vendor"] = report["cpu_vendor"]
        rung = LadderRung.from_mapping(raw)
        validate_role_capacity(rung, checkpoint, hardware_reports)
        with ThreadPoolExecutor(max_workers=min(startup_concurrency, len(workers))) as executor:
            list(
                executor.map(
                    lambda worker: start_worker(worker, remote_root, worker["name"]),
                    workers,
                )
            )
    except Exception:
        kill_workers(workers, revision)
        raise
    sdk = sdk_configuration(
        rung,
        checkpoint_path=checkpoint,
        fortress_seed=seed,
        certificate_path=output / "tls" / "client.crt",
        key_path=output / "tls" / "client.key",
        ca_path=output / "tls" / "ca.crt",
        role_packages=role_root,
    )
    if raw.get("datagram_tunnel") is True:
        start_datagram_tunnels(
            workers,
            remote_root,
            sdk,
            output / "tunnels",
            max_concurrency=startup_concurrency,
        )
    deployment = {
        "schema_version": 2,
        "source_revision": revision,
        "rung": raw,
        "placement": asdict(placement) if placement is not None else None,
        "hardware_validated": hardware_validated(hardware_reports, len(workers)),
        "hardware": hardware_reports,
        "workers": workers,
        "sdk": sdk,
        "recipient_key_directory": str(key_directory),
        "remote_root": remote_root,
    }
    destination = output / "deployment.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(deployment, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a Fortress ladder rung over SSH")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    deploy(args.manifest, args.output, skip_bootstrap=args.skip_bootstrap)


if __name__ == "__main__":
    main()
