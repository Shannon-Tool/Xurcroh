from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.config import ClusterConfig, ModelRegistry
from xurcroh.errors import ConfigurationError


def test_multiple_models_and_heterogeneous_cluster_normalize(tiny_model_path: Path) -> None:
    registry = ModelRegistry.normalize(
        {"qwen-7b": tiny_model_path, "llama-70b": str(tiny_model_path)}
    )
    assert set(registry.names()) == {"qwen-7b", "llama-70b"}
    cluster = ClusterConfig.normalize(
        [
            {
                "name": "worker-a",
                "host": "10.0.0.10",
                "ssh": {"username": "ubuntu", "key_path": "~/.ssh/id_ed25519"},
            },
            {
                "name": "worker-b",
                "endpoint": "quic://10.0.0.11:9443",
                "gpu": {"model": "H100", "memory_gb": 80},
            },
        ],
        pipeline_parallel=2,
    )
    assert len(cluster.workers) == 2
    assert cluster.workers[0].ssh is not None
    assert ".ssh/id_ed25519" in str(cluster.workers[0].ssh.key_path)


def test_minimal_cluster_defaults_to_explicit_local_cpu() -> None:
    cluster = ClusterConfig.normalize({})
    assert len(cluster.workers) == 1
    assert cluster.workers[0].endpoint == "local://cpu"
    assert cluster.startup_concurrency == 2


def test_single_worker_mapping_preserves_startup_concurrency() -> None:
    cluster = ClusterConfig.normalize({"host": "example.internal", "startup_concurrency": 1})
    assert cluster.startup_concurrency == 1


def test_pipeline_cannot_exceed_available_workers() -> None:
    with pytest.raises(ConfigurationError, match="pipeline_parallel"):
        ClusterConfig.normalize(["local://cpu"], pipeline_parallel=2)


def test_ssh_secrets_are_redacted() -> None:
    cluster = ClusterConfig.normalize(
        {"host": "example.internal", "ssh": {"username": "ubuntu", "password": "never-print-this"}}
    )
    rendered = repr(cluster)
    assert "never-print-this" not in rendered
    assert "**********" in rendered


def test_duplicate_model_names_are_rejected(tiny_model_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="invalid model name"):
        ModelRegistry.normalize({"bad name!": tiny_model_path})
