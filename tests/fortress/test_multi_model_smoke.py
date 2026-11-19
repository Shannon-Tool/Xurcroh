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
