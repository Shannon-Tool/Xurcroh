from __future__ import annotations

import json
from pathlib import Path

import yaml
from benchmarks.run import load_runtime_config, main


def test_benchmark_cli_runs_real_fwht_and_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "fwht.json"
    assert main(["fwht", "--dimension", "32", "--iterations", "2", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["kind"] == "fwht"
    assert report["maximum_inverse_error"] < 1e-12


def test_benchmark_config_preserves_advanced_cluster_settings(
    tmp_path: Path, tiny_model_path: Path
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "cluster": {"endpoint": "local://cpu"},
                "model_path": {"tiny": str(tiny_model_path)},
                "tensor_parallel": 1,
            }
        )
    )
    with load_runtime_config(config) as client:
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
    assert response.usage.completion_tokens == 1
