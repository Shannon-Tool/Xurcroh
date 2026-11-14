from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import ConfigurationError
from xurcroh.preflight import preflight_checkpoint


def test_preflight_reports_real_checkpoint_sizes_and_memory_estimates(
    tiny_llama_path: Path,
) -> None:
    definition = ModelDefinition(path=str(tiny_llama_path), engine="fortress")
    report = preflight_checkpoint(
        "tiny",
        definition,
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    assert report.checkpoint_bytes > 0
    assert report.estimated_client_peak_bytes > report.checkpoint_bytes
    assert report.estimated_role_worker_bytes > 0
    assert report.supported is True
    assert report.issues == ()


def test_preflight_rejects_insufficient_known_worker_memory(tiny_llama_path: Path) -> None:
    definition = ModelDefinition(path=str(tiny_llama_path), engine="fortress")
    report = preflight_checkpoint(
        "tiny",
        definition,
        [
            WorkerConfig(
                name="worker",
                endpoint="quic://worker:9444",
                gpu={"model": "tiny-gpu", "memory_gb": 0.000001},
                labels={"activation_index": "0", "weight_index": "0"},
            )
        ],
    )
    assert report.supported is False
    assert any("insufficient" in issue for issue in report.issues)


def test_preflight_rejects_non_fortress_remote_path_without_checkpoint(
    tmp_path: Path,
) -> None:
    definition = ModelDefinition(path=str(tmp_path / "missing"), engine="fortress")
    with pytest.raises(ConfigurationError, match="checkpoint"):
        preflight_checkpoint(
            "missing",
            definition,
            [WorkerConfig(name="worker", endpoint="quic://worker:9444")],
        )


def test_public_sdk_exposes_preflight_without_loading_model(tiny_llama_path: Path) -> None:
    with Xurcroh(
        model_path={"tiny": {"path": str(tiny_llama_path), "engine": "fortress"}}
    ) as client:
        reports = client.preflight()
        assert reports["tiny"].supported is True
        assert client.metrics()["requests_started"] == 0


def test_remote_fortress_requires_discovered_memory(tiny_llama_path: Path) -> None:
    definition = ModelDefinition(
        path=str(tiny_llama_path),
        revision="a" * 40,
        engine="fortress",
    )
    report = preflight_checkpoint(
        "tiny",
        definition,
        [WorkerConfig(name="worker", endpoint="quic://worker:9444")],
        require_known_memory=True,
    )
    assert report.supported is False
    assert any("memory is unknown" in issue for issue in report.issues)


def test_inference_fails_preflight_before_engine_creation(tiny_llama_path: Path) -> None:
    with (
        Xurcroh(
            cluster={
                "endpoint": "quic://worker:9444",
                "gpu": {"model": "tiny-gpu", "memory_gb": 0.000001},
            },
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "revision": "a" * 40,
                    "engine": "fortress",
                }
            },
        ) as client,
        pytest.raises(ConfigurationError, match="preflight failed"),
    ):
        client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
        )
