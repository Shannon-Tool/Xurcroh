from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Xurcroh import Xurcroh
from xurcroh.benchmarking import (
    benchmark_chat,
    benchmark_fwht,
    benchmark_transport,
    image_quality,
    write_report,
)


def test_chat_benchmark_measures_real_remote_inference(running_worker: str) -> None:
    with Xurcroh(
        cluster={"endpoint": running_worker, "security": {"mode": "development"}},
        model_path={"tiny": "remote-tiny"},
    ) as client:
        report = benchmark_chat(
            client,
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            iterations=2,
            max_tokens=2,
        )
    assert report["schema_version"] == 1
    assert report["environment"]["hardware_validated"] is False
    assert len(report["samples"]) == 2
    for sample in report["samples"]:
        assert sample["time_to_first_chunk_ms"] > 0
        assert sample["completion_tokens"] > 0
        assert sample["tokens_per_second"] > 0
    assert "hello" not in json.dumps(report)


def test_cpu_benchmarks_execute_real_algorithms() -> None:
    transform = benchmark_fwht(dimension=64, iterations=2, seed=7)
    assert transform["maximum_inverse_error"] < 1e-12
    assert transform["samples_per_second"] > 0
    transport = benchmark_transport(payload_bytes=256, iterations=2)
    assert transport["verified_round_trips"] == 2
    assert transport["aead_frames_per_second"] > 0


def test_image_quality_and_json_report(tmp_path: Path) -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = np.full((2, 2, 3), 3, dtype=np.uint8)
    quality = image_quality(reference, candidate)
    assert quality["mean_absolute_error"] == pytest.approx(3)
    assert quality["psnr_db"] == pytest.approx(38.5883785)
    path = write_report({"kind": "test", "metrics": quality}, tmp_path / "report.json")
    assert json.loads(path.read_text())["metrics"]["psnr_db"] == quality["psnr_db"]


def test_memory_profile_tracks_actual_cpu_allocations() -> None:
    from xurcroh.benchmarking import benchmark_memory

    report = benchmark_memory(dimension=64, rows=8, iterations=2, device="cpu")
    assert report["parameters"]["device"] == "cpu"
    assert report["peak_python_bytes"] > 0
    assert report["after"]["resident_megabytes"] > 0
    assert report["after"]["device"] == "cpu"


def test_media_benchmark_measures_real_neural_image() -> None:
    from models.test_diffusion_engine import TinyNeuralPipeline
    from xurcroh.benchmarking import benchmark_image
    from xurcroh.config import ModelDefinition
    from xurcroh.runtime.diffusion import LocalDiffusionEngine

    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={"tiny": {"path": "tiny", "modality": "image"}},
    ) as client:
        client._runtime._engines["tiny"] = LocalDiffusionEngine(
            "tiny",
            ModelDefinition(path="tiny", modality="image"),
            "local",
            pipeline=TinyNeuralPipeline(),
        )
        report = benchmark_image(client, model="tiny", prompt="city", size="8x8", steps=1, seed=1)
    assert report["encoded_bytes"] > 0
    assert report["duration_ms"] > 0
    assert report["pixels_per_second"] > 0
    assert "city" not in json.dumps(report)


def test_video_benchmark_decodes_output_and_compares_reference(tmp_path: Path) -> None:
    import base64

    from models.test_diffusion_engine import TinyNeuralPipeline
    from xurcroh.benchmarking import benchmark_video
    from xurcroh.config import ModelDefinition
    from xurcroh.runtime.diffusion import LocalDiffusionEngine

    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={"tiny": {"path": "tiny", "modality": "video"}},
    ) as client:
        client._runtime._engines["tiny"] = LocalDiffusionEngine(
            "tiny",
            ModelDefinition(path="tiny", modality="video"),
            "local",
            pipeline=TinyNeuralPipeline(),
        )
        response = client.videos.generate(
            model="tiny", prompt="waves", size="8x8", frames=2, fps=2, steps=1, seed=2
        )
        reference = tmp_path / "reference.mp4"
        reference.write_bytes(base64.b64decode(response.data[0].b64_json))
        report = benchmark_video(
            client,
            model="tiny",
            prompt="waves",
            size="8x8",
            frames=2,
            fps=2,
            steps=1,
            seed=2,
            reference=reference,
        )
    assert report["decoded_frames"] == 2
    assert report["generated_frames_per_second"] > 0
    quality = report["quality"]
    assert quality["mean_squared_error"] > 0
    assert quality["psnr_db"] is not None
    assert quality["psnr_db"] >= 20


def test_transport_benchmark_requires_native_extension(monkeypatch) -> None:
    monkeypatch.setattr("xurcroh.benchmarking._native_transport", lambda: None)
    with pytest.raises(RuntimeError, match="maturin develop"):
        benchmark_transport(iterations=1)
