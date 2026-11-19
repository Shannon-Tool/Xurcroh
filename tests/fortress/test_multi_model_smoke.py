from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multi_model_smoke_executes_real_models(tiny_llama_path: Path, tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "cluster": {"endpoint": "local://cpu"},
                "model_path": {
                    "tiny-a": {
                        "path": str(tiny_llama_path),
                        "engine": "fortress",
                        "fortress_seed": 731,
                    },
                    "tiny-b": {
                        "path": str(tiny_llama_path),
                        "engine": "fortress",
                        "fortress_seed": 732,
                    },
                },
            }
        )
    )
    output = tmp_path / "evidence.json"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(root / "scripts/fortress/multi_model_smoke.py"),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert set(report["models"]) == {"tiny-a", "tiny-b"}
    assert report["concurrent_requests"] == 2
    assert report["lifecycle"]["cancelled"] is True
    assert report["production_accepted"] is False


def test_public_sdk_executes_reap_pruned_model(tiny_llama_path: Path, tmp_path: Path) -> None:
    from tests.fortress.test_model_compiler import (
        test_moe_checkpoint_keeps_router_local_and_compiles_each_expert,
    )

    from Xurcroh import Xurcroh

    test_moe_checkpoint_keeps_router_local_and_compiles_each_expert(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "model_type": "qwen3_moe",
            "vocab_size": 12,
            "num_experts": 2,
            "moe_intermediate_size": 6,
            "intermediate_size": 6,
            "max_position_embeddings": 64,
        }
    )
    config_path.write_text(json.dumps(config))
    import shutil

    for source in tiny_llama_path.iterdir():
        if source.name not in {"config.json", "model.safetensors"} and source.is_file():
            shutil.copy(source, tmp_path / source.name)
    saliency = tmp_path / "saliency.json"
    saliency.write_text('{"0": {"0": 0.1, "1": 0.9}}')
    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={
            "moe": {
                "path": str(tmp_path),
                "engine": "fortress",
                "fortress_seed": 731,
                "reap_keep_fraction": 0.5,
                "reap_saliency_path": str(saliency),
            }
        },
    ) as client:
        response = client.chat.completions.create(
            model="moe",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=1,
        )
    assert response.usage.completion_tokens == 1
