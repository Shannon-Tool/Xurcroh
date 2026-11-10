from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_development_compose_is_loopback_only_and_explicitly_insecure() -> None:
    compose = yaml.safe_load((ROOT / "deploy/compose.yaml").read_text())
    assert compose["name"] == "xurcroh-development"
    for port in compose["services"]["controller"]["ports"]:
        assert port.startswith("127.0.0.1:")
    for worker in ("worker-a", "worker-b"):
        arguments = compose["services"][worker]["command"]
        assert "--development-http" in arguments
        assert any(value.startswith("--advertised-endpoint=") for value in arguments)
        assert compose["services"][worker]["read_only"] is True


def test_kubernetes_workers_require_tls_gpus_and_non_root_containers() -> None:
    documents = list(yaml.safe_load_all((ROOT / "deploy/kubernetes.yaml").read_text()))
    deployments = [item for item in documents if item["kind"] == "Deployment"]
    workers = [item for item in deployments if item["metadata"]["name"].startswith("worker-")]
    assert len(workers) == 2
    for worker in workers:
        spec = worker["spec"]["template"]["spec"]
        assert spec["securityContext"]["runAsNonRoot"] is True
        container = spec["containers"][0]
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
        assert "--development-http" not in container["args"]
        assert "--tls-cert=/tls/tls.crt" in container["args"]
        assert any(
            value.startswith("--advertised-endpoint=https://") for value in container["args"]
        )
    assert any(item["kind"] == "NetworkPolicy" for item in documents)


def test_container_build_uses_locked_dependencies_and_unprivileged_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "requirements.lock" in dockerfile
    assert "maturin build --release" in dockerfile
    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "USER 10001:10001" in dockerfile


def test_native_wheel_has_a_distinct_distribution_name() -> None:
    import tomllib

    sdk = tomllib.loads((ROOT / "pyproject.toml").read_text())
    native = tomllib.loads((ROOT / "crates/transport/pyproject.toml").read_text())
    assert native["project"]["name"] == "xurcroh-quic"
    assert native["project"]["name"] != sdk["project"]["name"]
    assert native["tool"]["maturin"]["module-name"] == "xurcroh_quic"
