from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from xurcroh.config import ClusterConfig, GPUConfig, ModelDefinition, WorkerConfig
from xurcroh.errors import ClusterUnavailableError, ConfigurationError
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.diffusers_loader import (
    compile_diffusers_component,
    discover_diffusers_operations,
)
from xurcroh.fortress.model_compiler import (
    compile_fortress_checkpoint,
    discover_fortress_operations,
)
from xurcroh.fortress.placement import (
    PlacementMap,
    PlacementPolicy,
    Provider,
    RoleAssignment,
    apply_placement,
    place_operations,
)
from xurcroh.fortress.tls import generate_mutual_tls
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker_cli import artifact_file_name
from xurcroh.preflight import detect_modality, ensure_checkpoint, preflight_checkpoint

REMOTE_ROOT = "/tmp"  # noqa: S108
GIT = shutil.which("git")


class RemoteExecutor(Protocol):
    def discover(self, worker: WorkerConfig) -> None: ...

    def run(
        self,
        worker: WorkerConfig,
        command: str,
        *,
        input_data: bytes | None = None,
        timeout: float | None = 120,
    ) -> str: ...

    def upload(
        self,
        worker: WorkerConfig,
        sources: list[Path],
        remote: str,
        *,
        recursive: bool = False,
    ) -> None: ...


class AsyncSSHExecutor:
    def _options(self, worker: WorkerConfig) -> dict[str, Any]:
        if worker.host is None or worker.ssh is None:
            raise ClusterUnavailableError(f"worker {worker.name!r} has no SSH configuration")
        ssh = worker.ssh
        options: dict[str, Any] = {
            "host": worker.host,
            "username": ssh.username,
            "port": ssh.port,
        }
        if ssh.key_path is not None:
            options["client_keys"] = [ssh.key_path]
        if ssh.password_env is not None:
            password = os.environ.get(ssh.password_env)
            if password is None:
                raise ClusterUnavailableError(
                    f"SSH credential variable {ssh.password_env} is not set"
                )
            options["password"] = password
        elif ssh.password is not None:
            options["password"] = ssh.password.get_secret_value()
        if ssh.known_hosts_path is not None:
            options["known_hosts"] = ssh.known_hosts_path
        if not ssh.strict_host_key_checking:
            options["known_hosts"] = None
        return options

    async def _run(
        self,
        worker: WorkerConfig,
        command: str,
        input_data: bytes | None,
        timeout: float | None,
    ) -> str:
        import asyncssh

        async with asyncssh.connect(**self._options(worker)) as connection:
            operation = connection.run(command, input=input_data, check=False, encoding=None)
            result = (
                await operation if timeout is None else await asyncio.wait_for(operation, timeout)
            )
        if result.exit_status not in (0, None):
            raise ClusterUnavailableError(
                f"remote command on {worker.name!r} failed with status {result.exit_status}"
            )
        output = result.stdout or ""
        return output.decode() if isinstance(output, bytes) else output

    def run(
        self,
        worker: WorkerConfig,
        command: str,
        *,
        input_data: bytes | None = None,
        timeout: float | None = 120,
    ) -> str:
        return asyncio.run(self._run(worker, command, input_data, timeout))

    def discover(self, worker: WorkerConfig) -> None:
        if worker.labels.get("device") == "cpu":
            worker.gpu = None
            return
        output = self.run(
            worker,
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
        )
        rows = [line for line in output.splitlines() if line.strip()]
        if not rows:
            raise ClusterUnavailableError(f"worker {worker.name!r} exposes no NVIDIA GPU")
        first_name, first_memory = rows[0].split(",", maxsplit=1)
        worker.gpu = GPUConfig(
            model=first_name.strip(),
            count=len(rows),
            memory_gb=float(first_memory.strip()) / 1024,
            identifiers=list(range(len(rows))),
        )

    async def _upload(
        self,
        worker: WorkerConfig,
        sources: list[Path],
        remote: str,
        recursive: bool,
    ) -> None:
        import asyncssh

        async with (
            asyncssh.connect(**self._options(worker)) as connection,
            connection.start_sftp_client() as sftp,
        ):
            await sftp.put([str(path) for path in sources], remote, recurse=recursive)

    def upload(
        self,
        worker: WorkerConfig,
        sources: list[Path],
        remote: str,
        *,
        recursive: bool = False,
    ) -> None:
        asyncio.run(self._upload(worker, sources, remote, recursive))


@dataclass(frozen=True)
class RoleDeployment:
    model: str
    model_index: int
    worker: WorkerConfig
    activation_index: int
    weight_index: int
    recipient_public: str
    recipient_key: Path
    package: Path
    service_name: str
    port_number: int

    @property
    def port(self) -> int:
        return self.port_number


class FortressDeployment:
    def __init__(
        self,
        cluster: ClusterConfig,
        models: tuple[tuple[str, ModelDefinition], ...],
        *,
        workspace: Path | None = None,
        executor_factory: Callable[[], RemoteExecutor] = AsyncSSHExecutor,
    ) -> None:
        self.cluster = cluster
        self.models = models
        self.executor_factory = executor_factory
        self._executor: RemoteExecutor | None = None
        self._workspace = workspace or Path(tempfile.mkdtemp(prefix="xurcroh-fortress-"))
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._deployment_id = uuid.uuid4().hex
        self._remote_root = f"{REMOTE_ROOT}/xurcroh-fortress-{self._deployment_id}"
        self._roles: list[RoleDeployment] = []
        self._workers: dict[str, list[WorkerConfig]] = {}
        self._placements: dict[str, PlacementMap] = {}
        self._tunnel_workers: list[list[dict[str, Any]]] = []
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self._restart_condition = threading.Condition(self._lock)
        self._restarting: set[str] = set()
        self._role_generations: dict[str, int] = {}

    @property
    def workers(self) -> Mapping[str, list[WorkerConfig]]:
        return self._workers

    @property
    def workspace(self) -> Path:
        return self._workspace

    def _validate(self) -> None:
        if not self.models:
            raise ConfigurationError("Fortress provisioning requires at least one model")
        for name, definition in self.models:
            if definition.engine != "fortress":
                raise ConfigurationError(
                    f"provision_workers supports protected Fortress models only; {name!r} uses "
                    f"{definition.engine!r}"
                )
            ensure_checkpoint(name, definition)
            detect_modality(definition)
        shares = self.models[0][1].fortress_shares
        minimum = 3 * shares**2
        if len(self.cluster.workers) < minimum:
            raise ConfigurationError(
                f"Fortress placement requires at least {minimum} workers for {shares}-by-"
                f"{shares} operation grids with adjacency and pair separation"
            )
        if any(
            definition.fortress_shares != self.models[0][1].fortress_shares
            for _, definition in self.models[1:]
        ):
            raise ConfigurationError(
                "all automatically provisioned models need the same share count"
            )
        for worker in self.cluster.workers:
            if worker.host is None or worker.ssh is None:
                raise ClusterUnavailableError(
                    f"worker {worker.name!r} requires host and SSH configuration for provisioning"
                )
            if not worker.provider.get("country") or not worker.provider.get("cpu_vendor"):
                raise ConfigurationError(
                    f"worker {worker.name!r} requires provider country and cpu_vendor metadata"
                )

    def _source_archive(self) -> bytes:
        if GIT is None:
            raise ConfigurationError("git is required for reproducible Fortress provisioning")
        root = Path(__file__).resolve().parents[3]
        status = subprocess.run(  # noqa: S603
            [GIT, "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise ConfigurationError(
                "commit all source changes before Fortress provisioning so workers run the "
                "same reviewed source revision"
            )
        revision = subprocess.run(  # noqa: S603
            [GIT, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return subprocess.run(  # noqa: S603
            [GIT, "archive", "--format=tar.gz", revision],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout

    def _write_secret(self, path: Path, value: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)

    def _place_workers(
        self, definition: ModelDefinition
    ) -> tuple[list[WorkerConfig], PlacementMap]:
        providers = [
            Provider(
                worker.name,
                str(worker.provider["country"]),
                str(worker.provider["cpu_vendor"]),
            )
            for worker in self.cluster.workers
        ]
        if definition.modality in {"image", "video"}:
            component = (
                "transformer" if (Path(definition.path) / "transformer").is_dir() else "unet"
            )
            operations = discover_diffusers_operations(definition.path, component=component)
        else:
            operations = discover_fortress_operations(definition.path)
        placement = place_operations(
            operations,
            providers,
            PlacementPolicy(
                shares=definition.fortress_shares,
                max_provider_fraction=definition.fortress_max_provider_fraction,
            ),
        )
        return apply_placement(self.cluster.workers, placement), placement

    def _create_roles(self) -> None:
        seed = int.from_bytes(os.urandom(32), "big")
        for model_index, (name, definition) in enumerate(self.models):
            model_root = self._workspace / f"model-{model_index}"
            model_root.mkdir()
            model_seed = definition.fortress_seed if definition.fortress_seed is not None else seed
            placed_workers, placement = self._place_workers(definition)
            placed_workers = [worker for worker in placed_workers if worker.labels["operations"]]
            self._placements[name] = placement
            assignments = {
                (assignment.provider_id, assignment.unit_id): assignment
                for assignment in placement.assignments
            }
            model_roles: list[RoleDeployment] = []
            for index, worker in enumerate(placed_workers):
                first_assignment = next(
                    assignment
                    for assignment in placement.assignments
                    if assignment.provider_id == worker.name
                )
                private = X25519PrivateKey.generate()
                public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                key_path = model_root / f"recipient-{index}.key"
                self._write_secret(
                    key_path,
                    private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                )
                model_roles.append(
                    RoleDeployment(
                        model=name,
                        model_index=model_index,
                        worker=worker,
                        activation_index=first_assignment.activation_share,
                        weight_index=first_assignment.weight_share,
                        recipient_public=public.hex(),
                        recipient_key=key_path,
                        package=model_root / f"role-{index}",
                        service_name=f"model-{model_index}-{worker.name}",
                        port_number=0,
                    )
                )
            self._roles.extend(model_roles)

            def sink(
                operation_name: str,
                operation: CompiledOperation,
                roles: list[RoleDeployment] = model_roles,
                assignments: dict[tuple[str, str], RoleAssignment] = assignments,
            ) -> None:
                for role in roles:
                    assignment = assignments.get((role.worker.name, operation_name))
                    if assignment is None:
                        continue
                    role.package.mkdir(parents=True, exist_ok=True)
                    encoded = encode_role_artifact(
                        operation,
                        assignment.weight_share,
                        bytes.fromhex(role.recipient_public),
                    )
                    self._write_secret(
                        role.package / f"{artifact_file_name(operation_name)}.json", encoded
                    )

            if definition.modality in {"image", "video"}:
                component = (
                    "transformer" if (Path(definition.path) / "transformer").is_dir() else "unet"
                )
                compile_diffusers_component(
                    definition.path,
                    component=component,
                    seed=model_seed,
                    share_count=definition.fortress_shares,
                    operation_sink=sink,
                )
            else:
                compile_fortress_checkpoint(
                    definition.path,
                    seed=model_seed,
                    share_count=definition.fortress_shares,
                    load_model=False,
                    operation_sink=sink,
                )
            if definition.fortress_seed is None:
                object.__setattr__(definition, "fortress_seed", model_seed)

    def _prepare_machine(self, worker: WorkerConfig, archive: bytes) -> None:
        assert self._executor is not None
        root = shlex.quote(self._remote_root)
        self._executor.run(worker, f"mkdir -p {root}/tls")
        self._executor.run(worker, f"tar -xzf - -C {root}", input_data=archive)
        if worker.labels.get("bootstrap", "true").lower() not in {"0", "false", "no"}:
            self._executor.run(
                worker,
                f"cd {root} && XURCROH_FORTRESS_ONLY=1 ./scripts/gpu/bootstrap.sh",
                timeout=3600,
            )

    def _prepare_role(self, role: RoleDeployment) -> None:
        assert self._executor is not None
        root = Path(self._remote_root)
        role_root = root / f"model-{role.model_index}" / role.worker.name
        remote = shlex.quote(str(role_root))
        self._executor.run(role.worker, f"mkdir -p {remote}/tls")
        self._executor.upload(
            role.worker, [role.package], str(role_root / "artifacts"), recursive=True
        )
        self._executor.upload(role.worker, [role.recipient_key], str(role_root / "recipient.key"))
        tls = self._workspace / "tls"
        self._executor.upload(
            role.worker,
            [tls / "ca.crt", tls / f"{role.service_name}.crt", tls / f"{role.service_name}.key"],
            str(role_root / "tls"),
        )
        self._executor.run(
            role.worker,
            f"chmod 600 {remote}/recipient.key {remote}/tls/{shlex.quote(role.service_name)}.key",
        )
        role.recipient_key.unlink()

    def _start_role(self, role: RoleDeployment) -> None:
        assert self._executor is not None
        root = Path(self._remote_root)
        role_root = root / f"model-{role.model_index}" / role.worker.name
        remote = shlex.quote(str(role_root))
        service = shlex.quote(role.service_name)
        ready = role_root / "ready"
        python = shlex.quote(role.worker.labels.get("python_path", ".venv-cuda/bin/python"))
        device = role.worker.labels.get("device", "cuda")
        if device not in {"cpu", "cuda"}:
            raise ConfigurationError(f"unsupported Fortress worker device: {device}")
        self._executor.run(
            role.worker,
            f"rm -f {shlex.quote(str(ready))} {shlex.quote(str(role_root / 'role.exit'))} "
            f"{shlex.quote(str(role_root / 'role.pid'))}",
        )
        command = (
            f"cd {shlex.quote(self._remote_root)} && setsid -f env HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 "
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:"
            f"/usr/local/cuda/lib64 {python} -m xurcroh.fortress.worker_cli "
            f"--artifact-dir {remote}/artifacts --recipient-key {remote}/recipient.key "
            f"--certificate {remote}/tls/{service}.crt --private-key {remote}/tls/{service}.key "
            f"--ca {remote}/tls/ca.crt --host 0.0.0.0 --port {role.port} --device {device} "
            f"--ready-file {shlex.quote(str(ready))} --exit-status-file {remote}/role.exit "
            f"--pid-file {shlex.quote(str(role_root / 'role.pid'))} "
            f"</dev/null >{remote}/role.log 2>&1"
        )
        logging.getLogger(__name__).info("Starting Fortress role %s", role.service_name)
        self._executor.run(role.worker, command, timeout=30)
        deadline = time.monotonic() + 120
        exit_file = shlex.quote(str(role_root / "role.exit"))
        while True:
            status = self._executor.run(
                role.worker,
                f"if test -f {shlex.quote(str(ready))}; then echo ready; "
                f"elif test -f {exit_file}; then echo exited; fi",
            ).strip()
            if status == "ready":
                address = self._executor.run(role.worker, f"cat {shlex.quote(str(ready))}").strip()
                try:
                    port = int(address.rsplit(":", maxsplit=1)[1])
                except (IndexError, ValueError) as error:
                    raise ClusterUnavailableError(
                        f"Fortress role {role.service_name!r} reported an invalid ready address"
                    ) from error
                if not 1 <= port <= 65535:
                    raise ClusterUnavailableError(
                        f"Fortress role {role.service_name!r} reported an invalid port"
                    )
                object.__setattr__(role, "port_number", port)
                return
            if status == "exited":
                try:
                    details = self._executor.run(
                        role.worker, f"tail -c 4000 {shlex.quote(str(role_root / 'role.log'))}"
                    ).strip()
                except Exception as log_error:
                    details = f"role.log could not be read: {log_error}"
                raise ClusterUnavailableError(
                    f"Fortress role {role.service_name!r} exited before readiness: {details}"
                )
            if time.monotonic() >= deadline:
                raise ClusterUnavailableError(
                    f"Fortress role {role.service_name!r} did not become ready; inspect role.log"
                )
            time.sleep(0.2)

    def _publish_workers(self) -> None:
        tls = self._workspace / "tls"
        for role in self._roles:
            worker = role.worker.model_copy(deep=True)
            worker.name = f"{role.worker.name}-{role.model}"
            worker.endpoint = f"quic://{role.worker.host}:{role.port}"
            worker.models = [role.model]
            worker.labels = {
                **worker.labels,
                "activation_index": str(role.activation_index),
                "weight_index": str(role.weight_index),
                "recipient_public": role.recipient_public,
                "client_certificate_path": str(tls / "client.crt"),
                "client_key_path": str(tls / "client.key"),
                "ca_path": str(tls / "ca.crt"),
                "server_name": role.service_name,
            }
            self._workers.setdefault(role.model, []).append(worker)

    def _tunnel_worker(self, worker: WorkerConfig) -> dict[str, Any]:
        assert worker.host is not None and worker.ssh is not None
        if worker.ssh.password is not None or worker.ssh.password_env is not None:
            raise ConfigurationError(
                "SSH datagram forwarding requires a key or agent, not password authentication"
            )
        return {
            "name": worker.name,
            "host": worker.host,
            "ssh_user": worker.ssh.username,
            "ssh_port": worker.ssh.port,
            "ssh_key": worker.ssh.key_path,
            "known_hosts": worker.ssh.known_hosts_path,
            "python_path": worker.labels.get("python_path", ".venv-cuda/bin/python"),
        }

    def _enable_tunnels(self) -> None:
        if not self.cluster.datagram_tunnel:
            return
        from xurcroh.fortress.datagram_deploy import start_datagram_tunnels

        for model_index, (name, _) in enumerate(self.models):
            generated = self._workers[name]
            tunnel_workers = [
                self._tunnel_worker(role.worker) for role in self._roles if role.model == name
            ]
            sdk = {"cluster": generated}
            start_datagram_tunnels(
                tunnel_workers,
                self._remote_root,
                sdk,
                self._workspace / "tunnels" / f"model-{model_index}",
                target_ports=[role.port for role in self._roles if role.model == name],
                tunnel_name=f"relay-{model_index}",
            )
            self._tunnel_workers.append(tunnel_workers)

    def start(self) -> FortressDeployment:
        with self._lock:
            if self._closed:
                raise ClusterUnavailableError("Fortress deployment is closed")
            if self._started:
                return self
            try:
                self._validate()
                self._executor = self.executor_factory()
                with ThreadPoolExecutor(max_workers=min(32, len(self.cluster.workers))) as pool:
                    list(pool.map(self._executor.discover, self.cluster.workers))
                for name, definition in self.models:
                    report = preflight_checkpoint(
                        name,
                        definition,
                        list(self.cluster.workers),
                        require_known_memory=True,
                        validate_roles=False,
                    )
                    if not report.supported:
                        raise ConfigurationError(
                            f"model {name!r} preflight failed: " + "; ".join(report.issues)
                        )
                self._create_roles()
                generate_mutual_tls(
                    self._workspace / "tls",
                    {role.service_name: [role.worker.host or ""] for role in self._roles},
                )
                archive = self._source_archive()
                with ThreadPoolExecutor(max_workers=min(32, len(self.cluster.workers))) as pool:
                    list(
                        pool.map(
                            lambda worker: self._prepare_machine(worker, archive),
                            self.cluster.workers,
                        )
                    )
                with ThreadPoolExecutor(max_workers=min(32, len(self._roles))) as pool:
                    list(pool.map(self._prepare_role, self._roles))
                with ThreadPoolExecutor(max_workers=min(32, len(self._roles))) as pool:
                    list(pool.map(self._start_role, self._roles))
                self._publish_workers()
                self._enable_tunnels()
                self._started = True
                return self
            except BaseException as error:
                try:
                    self.close()
                except Exception as cleanup_error:
                    error.add_note(f"Fortress cleanup also failed: {cleanup_error}")
                raise

    def _stop_role(self, role: RoleDeployment, timeout: float = 30) -> None:
        assert self._executor is not None
        role_root = Path(self._remote_root) / f"model-{role.model_index}" / role.worker.name
        pid = shlex.quote(str(role_root / "role.pid"))
        exit_file = shlex.quote(str(role_root / "role.exit"))
        self._executor.run(
            role.worker,
            f"if test -f {pid}; then kill $(cat {pid}) 2>/dev/null || true; fi",
        )
        deadline = time.monotonic() + timeout
        while True:
            status = self._executor.run(
                role.worker,
                f"if ! test -f {pid}; then echo exited; "
                f"elif test -f {exit_file}; then echo exited; "
                f"elif ! kill -0 $(cat {pid}) 2>/dev/null; then echo exited; fi",
            ).strip()
            if status == "exited":
                return
            if time.monotonic() >= deadline:
                self._executor.run(
                    role.worker,
                    f"if test -f {pid}; then kill -KILL $(cat {pid}) 2>/dev/null || true; fi",
                )
                raise ClusterUnavailableError(
                    f"Fortress role {role.service_name!r} required forced termination"
                )
            time.sleep(0.2)

    def diagnostics(self) -> dict[str, str]:
        if self._executor is None:
            return {}
        reports: dict[str, str] = {}
        for role in self._roles:
            path = (
                Path(self._remote_root)
                / f"model-{role.model_index}"
                / role.worker.name
                / "role.log"
            )
            try:
                output = self._executor.run(
                    role.worker, f"tail -c 4000 {shlex.quote(str(path))}", timeout=10
                ).strip()
                if output:
                    reports[role.service_name] = output
            except Exception as error:
                reports[role.service_name] = f"role.log unavailable: {error}"
        return reports

    def restart_role(self, provider_id: str, generation: int = 0) -> int:
        with self._restart_condition:
            current = self._role_generations.get(provider_id, 0)
            if current > generation:
                return current
            while provider_id in self._restarting:
                self._restart_condition.wait()
            current = self._role_generations.get(provider_id, 0)
            if current > generation:
                return current
            if self._closed or not self._started:
                raise ClusterUnavailableError(
                    f"cannot restart Fortress role {provider_id!r} before startup or after close"
                )
            matches = [
                role for role in self._roles if f"{role.worker.name}-{role.model}" == provider_id
            ]
            if len(matches) != 1:
                raise ClusterUnavailableError(f"unknown managed Fortress role: {provider_id}")
            role = matches[0]
            self._restarting.add(provider_id)
        try:
            self._stop_role(role)
            self._start_role(role)
        except BaseException:
            with self._restart_condition:
                self._restarting.remove(provider_id)
                self._restart_condition.notify_all()
            raise
        with self._restart_condition:
            current = self._role_generations.get(provider_id, 0) + 1
            self._role_generations[provider_id] = current
            self._restarting.remove(provider_id)
            self._restart_condition.notify_all()
            return current

    def close(self) -> None:
        with self._restart_condition:
            while self._restarting:
                self._restart_condition.wait()
            if self._closed and not self._workspace.exists():
                return
            self._closed = True
            executor = self._executor
            roles = list(self._roles)
        failures: list[str] = []
        failed_workers: set[str] = set()
        if executor is not None:
            from xurcroh.fortress.datagram_deploy import close_datagram_tunnels

            configured = {worker.name: worker for worker in self.cluster.workers}
            for group in self._tunnel_workers:
                try:
                    close_datagram_tunnels(group)
                except Exception as error:
                    failures.append(f"local relay cleanup failed: {error}")
                for tunneled in group:
                    tunnel = tunneled.get("datagram_tunnel", {})
                    pid_path = tunnel.get("remote_pid_path")
                    worker = configured.get(tunneled["name"])
                    if pid_path is None or worker is None:
                        continue
                    try:
                        quoted = shlex.quote(str(pid_path))
                        executor.run(
                            worker,
                            f"if test -f {quoted}; then kill $(cat {quoted}) "
                            "2>/dev/null || true; fi",
                        )
                    except Exception as error:
                        failures.append(f"{worker.name}: relay cleanup failed: {error}")
                        failed_workers.add(worker.name)
            for role in roles:
                try:
                    self._stop_role(role)
                except Exception as error:
                    failures.append(f"{role.service_name}: {error}")
                    failed_workers.add(role.worker.name)
            for worker in self.cluster.workers:
                if worker.name in failed_workers:
                    continue
                try:
                    executor.run(worker, f"rm -rf {shlex.quote(self._remote_root)}")
                except Exception as error:
                    failures.append(f"{worker.name}: remote cleanup failed: {error}")
        if failures:
            raise ClusterUnavailableError(
                "Fortress teardown incomplete; local recovery state retained at "
                f"{self._workspace}: " + "; ".join(failures)
            )
        shutil.rmtree(self._workspace)
