from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
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
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.tls import generate_mutual_tls
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker_cli import artifact_file_name
from xurcroh.preflight import ensure_checkpoint, preflight_checkpoint

REMOTE_ROOT = "/tmp"  # noqa: S108
GIT = shutil.which("git")


class RemoteExecutor(Protocol):
    def discover(self, worker: WorkerConfig) -> None: ...

    def run(
        self, worker: WorkerConfig, command: str, *, input_data: bytes | None = None
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

    async def _run(self, worker: WorkerConfig, command: str, input_data: bytes | None) -> str:
        import asyncssh

        async with asyncssh.connect(**self._options(worker)) as connection:
            result = await connection.run(command, input=input_data, check=False)
        if result.exit_status not in (0, None):
            raise ClusterUnavailableError(
                f"remote command on {worker.name!r} failed with status {result.exit_status}"
            )
        output = result.stdout or ""
        return output.decode() if isinstance(output, bytes) else output

    def run(self, worker: WorkerConfig, command: str, *, input_data: bytes | None = None) -> str:
        return asyncio.run(self._run(worker, command, input_data))

    def discover(self, worker: WorkerConfig) -> None:
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

    @property
    def port(self) -> int:
        return 9444 + self.model_index


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
        self._started = False
        self._closed = False
        self._lock = threading.RLock()

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
            if definition.modality not in {"auto", "text", "vlm"}:
                raise ConfigurationError(
                    f"automatic Fortress provisioning does not support {definition.modality!r} "
                    f"model {name!r}; deploy its diffusion roles explicitly"
                )
            ensure_checkpoint(name, definition)
        expected = 2 * self.models[0][1].fortress_shares
        if len(self.cluster.workers) != expected:
            raise ConfigurationError(
                f"Fortress provisioning requires exactly {expected} workers for one complete "
                f"2-by-{self.models[0][1].fortress_shares} role grid"
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

    def _source_archive(self) -> bytes:
        if GIT is None:
            raise ConfigurationError("git is required for reproducible Fortress provisioning")
        root = Path(__file__).resolve().parents[3]
        status = subprocess.run(  # noqa: S603
            [GIT, "status", "--porcelain"],
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

    def _create_roles(self) -> None:
        seed = int.from_bytes(os.urandom(32), "big")
        for model_index, (name, definition) in enumerate(self.models):
            model_root = self._workspace / f"model-{model_index}"
            model_root.mkdir()
            model_seed = definition.fortress_seed if definition.fortress_seed is not None else seed
            model_roles: list[RoleDeployment] = []
            for index, worker in enumerate(self.cluster.workers):
                activation_index, weight_index = divmod(index, definition.fortress_shares)
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
                        activation_index=activation_index,
                        weight_index=weight_index,
                        recipient_public=public.hex(),
                        recipient_key=key_path,
                        package=model_root / f"role-{index}",
                        service_name=f"model-{model_index}-{worker.name}",
                    )
                )
            self._roles.extend(model_roles)

            def sink(
                operation_name: str,
                operation: CompiledOperation,
                roles: list[RoleDeployment] = model_roles,
            ) -> None:
                for role in roles:
                    role.package.mkdir(parents=True, exist_ok=True)
                    encoded = encode_role_artifact(
                        operation, role.weight_index, bytes.fromhex(role.recipient_public)
                    )
                    self._write_secret(
                        role.package / f"{artifact_file_name(operation_name)}.json", encoded
                    )

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
        self._executor.run(worker, f"cd {root} && ./scripts/gpu/bootstrap.sh")

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

    def _start_role(self, role: RoleDeployment) -> None:
        assert self._executor is not None
        root = Path(self._remote_root)
        role_root = root / f"model-{role.model_index}" / role.worker.name
        remote = shlex.quote(str(role_root))
        service = shlex.quote(role.service_name)
        ready = role_root / "ready"
        command = (
            f"cd {shlex.quote(self._remote_root)} && setsid env HF_HUB_OFFLINE=1 "
            "TRANSFORMERS_OFFLINE=1 "
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:"
            "/usr/local/cuda/lib64 .venv-cuda/bin/python -m xurcroh.fortress.worker_cli "
            f"--artifact-dir {remote}/artifacts --recipient-key {remote}/recipient.key "
            f"--certificate {remote}/tls/{service}.crt --private-key {remote}/tls/{service}.key "
            f"--ca {remote}/tls/ca.crt --host 0.0.0.0 --port {role.port} --device cuda "
            f"--ready-file {shlex.quote(str(ready))} --exit-status-file {remote}/role.exit "
            f"</dev/null >{remote}/role.log 2>&1 & echo $! >{remote}/role.pid"
        )
        self._executor.run(role.worker, command)
        deadline = time.monotonic() + 120
        exit_file = shlex.quote(str(role_root / "role.exit"))
        while True:
            status = self._executor.run(
                role.worker,
                f"if test -f {shlex.quote(str(ready))}; then echo ready; "
                f"elif test -f {exit_file}; then echo exited; fi",
            ).strip()
            if status == "ready":
                return
            if status == "exited":
                raise ClusterUnavailableError(
                    f"Fortress role {role.service_name!r} exited before readiness; inspect role.log"
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

    def start(self) -> FortressDeployment:
        with self._lock:
            if self._closed:
                raise ClusterUnavailableError("Fortress deployment is closed")
            if self._started:
                return self
            try:
                self._validate()
                self._executor = self.executor_factory()
                for worker in self.cluster.workers:
                    self._executor.discover(worker)
                for name, definition in self.models:
                    report = preflight_checkpoint(
                        name,
                        definition,
                        list(self.cluster.workers),
                        require_known_memory=True,
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
                for worker in self.cluster.workers:
                    self._prepare_machine(worker, archive)
                for role in self._roles:
                    self._prepare_role(role)
                for role in self._roles:
                    self._start_role(role)
                self._publish_workers()
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
                f"if test -f {exit_file}; then echo exited; "
                f"elif test -f {pid} && ! kill -0 $(cat {pid}) 2>/dev/null; "
                "then echo exited; fi",
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

    def close(self) -> None:
        with self._lock:
            if self._closed and not self._workspace.exists():
                return
            self._closed = True
            executor = self._executor
            roles = list(self._roles)
        failures: list[str] = []
        failed_workers: set[str] = set()
        if executor is not None:
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
