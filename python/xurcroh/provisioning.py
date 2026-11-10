from __future__ import annotations

import asyncio
import os
import shlex
import threading
from typing import Any

from xurcroh.config import ClusterConfig, GPUConfig, ModelDefinition, WorkerConfig
from xurcroh.errors import ClusterUnavailableError, CompatibilityError


def build_worker_command(
    worker: WorkerConfig,
    *,
    model_paths: dict[str, str],
    controller: str,
) -> str:
    arguments = ["python3", "-m", "xurcroh.worker", "--worker-id", worker.name]
    for name, path in sorted(model_paths.items()):
        arguments.extend(["--model", f"{name}={path}"])
    arguments.extend(["--controller", controller])
    return " ".join(shlex.quote(argument) for argument in arguments)


class SSHProvisioner:
    def _connection_options(self, worker: WorkerConfig) -> dict[str, Any]:
        if worker.host is None or worker.ssh is None:
            raise ClusterUnavailableError(f"worker {worker.name} has no SSH configuration")
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

    async def discover(self, worker: WorkerConfig) -> GPUConfig | None:
        try:
            import asyncssh
        except ImportError as error:
            raise CompatibilityError(
                "SSH discovery requires the ssh optional dependency"
            ) from error
        options = self._connection_options(worker)
        async with asyncssh.connect(**options) as connection:
            result = await connection.run(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                check=True,
            )
        output = result.stdout or ""
        if isinstance(output, bytes):
            output = output.decode()
        lines = [line for line in output.splitlines() if line.strip()]
        if not lines:
            return None
        name, memory = lines[0].split(",", maxsplit=1)
        return GPUConfig(
            model=name.strip(),
            count=len(lines),
            memory_gb=float(memory.strip()) / 1024,
            identifiers=list(range(len(lines))),
        )

    async def bootstrap(
        self,
        worker: WorkerConfig,
        *,
        model_paths: dict[str, str],
        controller: str,
    ) -> None:
        try:
            import asyncssh
        except ImportError as error:
            raise CompatibilityError(
                "SSH bootstrap requires the ssh optional dependency"
            ) from error
        options = self._connection_options(worker)
        command = build_worker_command(worker, model_paths=model_paths, controller=controller)
        async with asyncssh.connect(**options) as connection:
            result = await connection.run(f"nohup {command} >/dev/null 2>&1 &", check=False)
        if result.exit_status not in (0, None):
            raise ClusterUnavailableError(
                f"worker {worker.name} bootstrap failed with status {result.exit_status}"
            )


async def provision_cluster(
    cluster: ClusterConfig,
    models: tuple[tuple[str, ModelDefinition], ...],
    *,
    controller: str,
) -> None:
    provisioner = SSHProvisioner()
    for worker in cluster.workers:
        if worker.ssh is None:
            if worker.endpoint is None:
                raise ClusterUnavailableError(
                    f"worker {worker.name} cannot be provisioned without SSH"
                )
            continue
        gpu = await provisioner.discover(worker)
        if gpu is not None:
            worker.gpu = gpu
        await provisioner.bootstrap(
            worker,
            model_paths={name: definition.path for name, definition in models},
            controller=controller,
        )


def provision_cluster_sync(
    cluster: ClusterConfig,
    models: tuple[tuple[str, ModelDefinition], ...],
    *,
    controller: str,
) -> None:
    failure: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(provision_cluster(cluster, models, controller=controller))
        except BaseException as error:
            failure.append(error)

    thread = threading.Thread(target=run, name="xurcroh-provision")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
