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
        self.fail_cleanup = False

    def discover(self, worker: WorkerConfig) -> None:
        worker.gpu = {"model": "test-gpu", "memory_gb": 80}

    def run(
        self,
        worker: WorkerConfig,
        command: str,
        *,
        input_data: bytes | None = None,
        timeout: float | None = 120,
    ) -> str:
        del input_data, timeout
        self.commands.append((worker.name, command))
        if self.fail_cleanup and "rm -rf" in command:
            raise ClusterUnavailableError("SSH connection lost")
        if "role.exit" in command and "ready" not in command:
            return "exited\n"
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


def provisioned_cluster(count: int = 20) -> ClusterConfig:
    return ClusterConfig.normalize(
        {
            "workers": [
                {
                    "name": f"worker-{index}",
                    "host": f"10.0.0.{10 + index}",
                    "ssh": {"username": "ubuntu"},
                    "provider": {
                        "country": "us" if index % 2 else "de",
                        "cpu_vendor": "amd" if index % 2 else "intel",
                    },
                }
                for index in range(count)
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
    assert len(deployment.workers["tiny"]) == 20
    assert len({worker.labels["recipient_public"] for worker in deployment.workers["tiny"]}) == 20
    assert all(worker.labels["operation_roles"] for worker in deployment.workers["tiny"])
    assert len(deployment._placements["tiny"].assignments) == 14 * 4
    assert all(role.package.is_dir() for role in deployment._roles)
    workspace = deployment.workspace
    assert not (workspace / "tls" / "ca.key").exists()
    assert (workspace / "tls" / "client.key").is_file()
    assert all(not role.recipient_key.exists() for role in deployment._roles)
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
    with pytest.raises(ConfigurationError, match="at least 12 workers"):
        deployment.start()


def test_provisioning_accepts_complete_three_share_role_grid(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = provisioned_cluster(42)
    deployment = FortressDeployment(
        cluster,
        (
            (
                "tiny",
                ModelDefinition(
                    path=str(tiny_llama_path),
                    revision="a" * 40,
                    engine="fortress",
                    fortress_shares=3,
                ),
            ),
        ),
        workspace=tmp_path / "deployment",
        executor_factory=RecordingExecutor,
    )

    deployment._validate()
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    deployment.start()
    try:
        roles = deployment.workers["tiny"]
        assert len(roles) == 42
        placement = deployment._placements["tiny"]
        assert len(placement.assignments) == 14 * 9
        for operation in placement.operations:
            assigned = [
                assignment
                for assignment in placement.assignments
                if assignment.unit_id == operation.identifier
            ]
            assert len({(item.activation_share, item.weight_share) for item in assigned}) == 9
    finally:
        deployment.close()


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


def test_concurrent_role_recovery_restarts_each_role_once(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    deployment = FortressDeployment(
        provisioned_cluster(),
        (("tiny", ModelDefinition(path=str(tiny_llama_path), engine="fortress")),),
        workspace=tmp_path / "deployment",
        executor_factory=RecordingExecutor,
    )
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    deployment.start()
    role = deployment._roles[0]
    provider_id = f"{role.worker.name}-{role.model}"
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def stop(role) -> None:
        calls.append("stop")
        started.set()
        assert release.wait(timeout=5)

    def start(role) -> None:
        calls.append("start")

    monkeypatch.setattr(deployment, "_stop_role", stop)
    monkeypatch.setattr(deployment, "_start_role", start)
    waiting = threading.Event()
    original_wait = deployment._restart_condition.wait

    def wait() -> bool:
        waiting.set()
        return original_wait()

    monkeypatch.setattr(deployment._restart_condition, "wait", wait)
    threads = [
        threading.Thread(target=deployment.restart_role, args=(provider_id,)) for _ in range(2)
    ]
    try:
        threads[0].start()
        assert started.wait(timeout=5)
        threads[1].start()
        assert waiting.wait(timeout=5)
        release.set()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert calls == ["stop", "start"]
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
        deployment.close()


def test_failed_remote_cleanup_retains_recovery_state_and_can_retry(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    executor = RecordingExecutor()
    deployment = FortressDeployment(
        provisioned_cluster(),
        (("tiny", ModelDefinition(path=str(tiny_llama_path), engine="fortress")),),
        workspace=tmp_path / "deployment",
        executor_factory=lambda: executor,
    )
    deployment._executor = executor
    executor.fail_cleanup = True
    with pytest.raises(ClusterUnavailableError, match="recovery state retained"):
        deployment.close()
    assert deployment.workspace.exists()
    executor.fail_cleanup = False
    deployment.close()
    assert not deployment.workspace.exists()


def test_automatic_media_provisioning_uses_diffusers_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        norm_num_groups=4,
    )
    checkpoint = tmp_path / "model"
    diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler()).save_pretrained(checkpoint)
    definition = ModelDefinition(path=str(checkpoint), engine="fortress", fortress_seed=731)
    executor = RecordingExecutor()
    deployment = FortressDeployment(
        provisioned_cluster(24),
        (("tiny-image", definition),),
        workspace=tmp_path / "deployment",
        executor_factory=lambda: executor,
    )
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    deployment.start()
    assert definition.modality == "image"
    assert deployment._roles[0].package.is_dir()
    assert any(deployment._roles[0].package.iterdir())
    assert deployment.workers["tiny-image"][0].endpoint == "quic://10.0.0.10:9444"
    deployment.close()


def test_managed_datagram_tunnel_rewrites_endpoints_and_cleans_relay(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = RecordingExecutor()
    cluster = provisioned_cluster().model_copy(update={"datagram_tunnel": True})
    definition = ModelDefinition(path=str(tiny_llama_path), engine="fortress", fortress_seed=731)
    deployment = FortressDeployment(
        cluster,
        (("tiny", definition),),
        workspace=tmp_path / "deployment",
        executor_factory=lambda: executor,
    )
    monkeypatch.setattr(deployment, "_source_archive", lambda: b"source-archive")
    closed = []

    def start(
        workers, remote_root, sdk, output, *, target_port=9444, target_ports=None, tunnel_name
    ):
        assert target_port == 9444
        assert target_ports == [9444] * 20
        assert tunnel_name == "relay-0"
        for worker, configured in zip(workers, sdk["cluster"], strict=True):
            worker["datagram_tunnel"] = {
                "local_relay_pid": 99999999,
                "ssh_forward_pid": 99999998,
                "remote_pid_path": f"{remote_root}/{tunnel_name}-0.pid",
            }
            configured.endpoint = "quic://127.0.0.1:12000"

    monkeypatch.setattr("xurcroh.fortress.datagram_deploy.start_datagram_tunnels", start)
    monkeypatch.setattr(
        "xurcroh.fortress.datagram_deploy.close_datagram_tunnels",
        lambda workers: closed.append(workers),
    )
    deployment.start()
    assert deployment.workers["tiny"][0].endpoint == "quic://127.0.0.1:12000"
    deployment.close()
    assert len(closed) == 1
    assert any("relay-0-0.pid" in command for _, command in executor.commands)
