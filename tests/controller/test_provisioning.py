from __future__ import annotations

import types
from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.config import ClusterConfig, ModelDefinition, WorkerConfig
from xurcroh.errors import ClusterUnavailableError, ConfigurationError
from xurcroh.fortress.deployment import AsyncSSHExecutor, FortressDeployment


class RecordingExecutor:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, tuple[str, ...], str]] = []

    def discover(self, worker: WorkerConfig) -> None:
        worker.gpu = {"model": "test-gpu", "memory_gb": 80}

    def run(self, worker: WorkerConfig, command: str, *, input_data: bytes | None = None) -> str:
        del input_data
        self.commands.append((worker.name, command))
        return "ready\n" if "test -f" in command else ""

    def upload(
        self,
        worker: WorkerConfig,
        sources: list[Path],
        remote: str,
        *,
        recursive: bool = False,
    ) -> None:
        del recursive
        self.uploads.append((worker.name, tuple(str(path) for path in sources), remote))


def provisioned_cluster() -> ClusterConfig:
    return ClusterConfig.normalize(
        {
            "workers": [
                {
                    "name": f"worker-{index}",
                    "host": f"10.0.0.{10 + index}",
                    "ssh": {"username": "ubuntu"},
                }
                for index in range(4)
            ],
            "provision_workers": True,
            "controller_endpoint": "local://controller",
        }
    )


def test_fortress_provisioning_deploys_encrypted_roles_and_cleans_up(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = RecordingExecutor()
    definition = ModelDefinition(
        path=str(tiny_llama_path),
        revision="a" * 40,
        engine="fortress",
        fortress_seed=731,
    )
    deployment = FortressDeployment(
        provisioned_cluster(),
        (("tiny", definition),),
        workspace=tmp_path / "deployment",
        executor_factory=lambda: executor,
    )
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    deployment.start()
    assert len(deployment.workers["tiny"]) == 4
    assert len({worker.labels["recipient_public"] for worker in deployment.workers["tiny"]}) == 4
    assert [worker.labels["activation_index"] for worker in deployment.workers["tiny"]] == [
        "0",
        "0",
        "1",
        "1",
    ]
    workspace = deployment.workspace
    deployment.close()
    deployment.close()
    commands = [command for _, command in executor.commands]
    assert any("xurcroh.fortress.worker_cli" in command for command in commands)
    assert all("python3 -m xurcroh.worker" not in command for command in commands)
    assert not workspace.exists()
    assert any("rm -rf" in command for command in commands)


def test_provisioning_rejects_plaintext_models(tiny_llama_path: Path, tmp_path: Path) -> None:
    deployment = FortressDeployment(
        provisioned_cluster(),
        (("tiny", ModelDefinition(path=str(tiny_llama_path), engine="vllm")),),
        workspace=tmp_path / "deployment",
        executor_factory=RecordingExecutor,
    )
    with pytest.raises(ConfigurationError, match="protected Fortress models only"):
        deployment.start()
    assert not (tmp_path / "deployment").exists()


def test_provisioning_requires_complete_role_grid(tiny_llama_path: Path, tmp_path: Path) -> None:
    cluster = ClusterConfig.normalize(
        {
            "host": "worker",
            "ssh": {"username": "ubuntu"},
            "provision_workers": True,
            "controller_endpoint": "local://controller",
        }
    )
    deployment = FortressDeployment(
        cluster,
        (
            (
                "tiny",
                ModelDefinition(path=str(tiny_llama_path), revision="a" * 40, engine="fortress"),
            ),
        ),
        workspace=tmp_path / "deployment",
        executor_factory=RecordingExecutor,
    )
    with pytest.raises(ConfigurationError, match="exactly 4 workers"):
        deployment.start()


def test_asyncssh_executor_scopes_credentials_and_validates_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    connections = []

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, command, **kwargs):
            connections.append((command, kwargs))
            return types.SimpleNamespace(exit_status=0, stdout=b"output")

    def connect(**options):
        connections.append(options)
        return Connection()

    module = types.ModuleType("asyncssh")
    module.connect = connect
    monkeypatch.setitem(sys.modules, "asyncssh", module)
    monkeypatch.setenv("TEST_SSH_PASSWORD", "not-logged")
    worker = WorkerConfig(
        name="gpu-worker",
        host="worker.example",
        ssh={
            "username": "ubuntu",
            "port": 2222,
            "key_path": str(tmp_path / "identity"),
            "password_env": "TEST_SSH_PASSWORD",
            "strict_host_key_checking": False,
        },
    )
    executor = AsyncSSHExecutor()
    assert executor.run(worker, "true") == "output"
    assert connections[0]["host"] == "worker.example"
    assert connections[0]["known_hosts"] is None
    assert connections[0]["password"] == "not-logged"
    assert connections[0]["client_keys"] == [str(tmp_path / "identity")]
    assert "not-logged" not in connections[1][0]


def test_asyncssh_executor_reports_remote_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, command, **kwargs):
            return types.SimpleNamespace(exit_status=17, stdout="")

    module = types.ModuleType("asyncssh")
    module.connect = lambda **options: Connection()
    monkeypatch.setitem(sys.modules, "asyncssh", module)
    worker = WorkerConfig(
        name="remote", host="remote", ssh={"username": "ubuntu", "password_env": "MISSING"}
    )
    monkeypatch.delenv("MISSING", raising=False)
    with pytest.raises(ClusterUnavailableError, match="MISSING"):
        AsyncSSHExecutor().run(worker, "true")
    monkeypatch.setenv("MISSING", "secret")
    with pytest.raises(ClusterUnavailableError, match="status 17"):
        AsyncSSHExecutor().run(worker, "true")


def test_public_sdk_owns_provisioned_fortress_lifecycle(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = RecordingExecutor()
    cluster = provisioned_cluster()
    definition = ModelDefinition(
        path=str(tiny_llama_path),
        revision="a" * 40,
        engine="fortress",
        fortress_seed=731,
    )
    deployment = FortressDeployment(
        cluster,
        (("tiny", definition),),
        workspace=tmp_path / "deployment",
        executor_factory=lambda: executor,
    )
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    deployment.start()
    monkeypatch.setattr(
        "xurcroh.runtime.service.provision_cluster_sync", lambda *args, **kwargs: deployment
    )
    client = Xurcroh(
        cluster={
            "workers": [
                {"name": worker.name, "host": worker.host, "ssh": {"username": "ubuntu"}}
                for worker in cluster.workers
            ]
        },
        model_path={
            "tiny": {
                "path": str(tiny_llama_path),
                "revision": "a" * 40,
                "engine": "fortress",
                "fortress_seed": 731,
            }
        },
    )
    assert client.preflight("tiny")["tiny"].supported is True
    assert client._runtime._workers_for_model("tiny")[0].endpoint == "quic://10.0.0.10:9444"
    workspace = deployment.workspace
    client.close()
    assert not workspace.exists()
