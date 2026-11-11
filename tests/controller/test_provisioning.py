from __future__ import annotations

from xurcroh.config import WorkerConfig
from xurcroh.provisioning import build_worker_command


def test_worker_command_is_shell_quoted_and_contains_no_credentials() -> None:
    worker = WorkerConfig(
        name="worker-a",
        host="10.0.0.10",
        ssh={"username": "ubuntu", "key_path": "/private/key", "password": "secret"},
    )
    command = build_worker_command(
        worker,
        model_paths={"tiny": "/models/tiny"},
        controller="https://controller.internal:8443",
    )
    assert "secret" not in command
    assert "/private/key" not in command
    assert "worker-a" in command
    assert "/models/tiny" in command


async def test_ssh_discovery_and_bootstrap_use_scoped_connection_options(
    monkeypatch, tmp_path
) -> None:
    import sys
    import types

    from xurcroh.provisioning import SSHProvisioner

    connections = []

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, command, check):
            connections.append((command, check))
            if command.startswith("nvidia-smi"):
                return types.SimpleNamespace(stdout=b"GPU A, 40960\nGPU B, 40960\n")
            return types.SimpleNamespace(exit_status=0)

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
            "known_hosts_path": str(tmp_path / "known_hosts"),
            "strict_host_key_checking": False,
        },
    )
    provisioner = SSHProvisioner()
    gpu = await provisioner.discover(worker)
    assert gpu is not None
    assert gpu.count == 2
    assert gpu.memory_gb == 40
    assert gpu.identifiers == [0, 1]
    await provisioner.bootstrap(
        worker, model_paths={"model name": "/models/a b"}, controller="https://controller"
    )
    assert connections[0]["host"] == "worker.example"
    assert connections[0]["known_hosts"] is None
    assert connections[0]["password"] == "not-logged"
    assert connections[0]["client_keys"] == [str(tmp_path / "identity")]
    assert "'model name=/models/a b'" in connections[3][0]
    assert "not-logged" not in connections[3][0]


async def test_ssh_failure_paths(monkeypatch) -> None:
    import sys
    import types

    import pytest

    from xurcroh.errors import ClusterUnavailableError
    from xurcroh.provisioning import SSHProvisioner

    provisioner = SSHProvisioner()
    with pytest.raises(ClusterUnavailableError, match="no SSH configuration"):
        provisioner._connection_options(WorkerConfig(name="remote", endpoint="https://remote"))
    worker = WorkerConfig(
        name="remote", host="remote", ssh={"username": "ubuntu", "password_env": "MISSING_PASSWORD"}
    )
    monkeypatch.delenv("MISSING_PASSWORD", raising=False)
    with pytest.raises(ClusterUnavailableError, match="MISSING_PASSWORD"):
        provisioner._connection_options(worker)

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, command, check):
            return types.SimpleNamespace(exit_status=17, stdout="")

    module = types.ModuleType("asyncssh")
    module.connect = lambda **options: Connection()
    monkeypatch.setitem(sys.modules, "asyncssh", module)
    monkeypatch.setattr(provisioner, "_connection_options", lambda worker: {})
    assert await provisioner.discover(worker) is None
    with pytest.raises(ClusterUnavailableError, match="status 17"):
        await provisioner.bootstrap(worker, model_paths={}, controller="controller")


async def test_cluster_provisioning_and_sync_error_propagation(monkeypatch) -> None:
    import pytest

    from xurcroh.config import ClusterConfig, ModelDefinition
    from xurcroh.provisioning import SSHProvisioner, provision_cluster, provision_cluster_sync

    calls = []

    async def discover(self, worker):
        calls.append(("discover", worker.name))
        return None

    async def bootstrap(self, worker, *, model_paths, controller):
        calls.append(("bootstrap", worker.name, model_paths, controller))

    monkeypatch.setattr(SSHProvisioner, "discover", discover)
    monkeypatch.setattr(SSHProvisioner, "bootstrap", bootstrap)
    cluster = ClusterConfig.normalize(
        [
            {"name": "ssh", "host": "remote", "ssh": {"username": "ubuntu"}},
            {"name": "existing", "endpoint": "https://worker"},
        ]
    )
    models = (("tiny", ModelDefinition(path="/models/tiny")),)
    await provision_cluster(cluster, models, controller="https://controller")
    assert [call[:2] for call in calls] == [("discover", "ssh"), ("bootstrap", "ssh")]
    provision_cluster_sync(cluster, models, controller="https://controller")

    invalid = ClusterConfig.normalize({"name": "unprovisionable", "host": "remote"})
    with pytest.raises(Exception, match="cannot be provisioned without SSH"):
        await provision_cluster(invalid, models, controller="controller")

    async def fail(*args, **kwargs):
        raise RuntimeError("provision failed")

    monkeypatch.setattr("xurcroh.provisioning.provision_cluster", fail)
    with pytest.raises(RuntimeError, match="provision failed"):
        provision_cluster_sync(cluster, models, controller="controller")
