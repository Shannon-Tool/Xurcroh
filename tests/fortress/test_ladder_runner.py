from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tests.fortress.test_ladder import evidence_mapping, manifest_mapping

from xurcroh.fortress.ladder import verify_rung_evidence


def test_ladder_cli_entry_points_execute_directly() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("run_ladder.py", "deploy_ladder.py", "cuda_baseline.py", "grid_smoke.py"):
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(root / "scripts/fortress" / name), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_ladder_runner_writes_verified_signed_evidence(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts/fortress/run_ladder.py"
    spec = importlib.util.spec_from_file_location("run_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_mapping()))
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps({"source_revision": "a" * 40, "workers": [], "sdk": {}}))
    private = Ed25519PrivateKey.generate()
    signing_key = tmp_path / "signing.key"
    signing_key.write_bytes(private.private_bytes_raw())
    report = evidence_mapping()
    del report["worker_exit_codes"]
    del report["cost_usd"]
    del report["destruction"]
    module.verify_ladder = lambda deployment: report
    module.stop_workers = lambda workers, revision: [0, 0, 0, 0]
    module.main(
        [
            str(manifest),
            "--existing-deployment",
            str(deployment),
            "--output",
            str(tmp_path / "evidence"),
            "--signing-key",
            str(signing_key),
            "--uncapped",
        ]
    )
    rung, verified = verify_rung_evidence(
        tmp_path / "evidence" / "evidence.json", public_key=private.public_key()
    )
    assert rung.rung == "L0"
    assert verified["worker_exit_codes"] == [0, 0, 0, 0]
    assert verified["cost_usd"] == 0
    assert verified["source_revision"] == "a" * 40
