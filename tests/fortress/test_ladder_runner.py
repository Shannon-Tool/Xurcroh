from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.fortress.test_ladder import manifest_mapping
from tests.fortress.test_ladder_v2 import valid_v2_report

from xurcroh.fortress.ladder import LadderRung, verify_rung_evidence, write_rung_evidence
from xurcroh.signatures import HybridSigningKey


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
    private = HybridSigningKey.generate()
    signing_key = tmp_path / "signing.key"
    signing_key.write_bytes(private.to_bytes())
    previous_report = valid_v2_report()
    previous_report["hardware_validated"] = False
    previous_report["source_revision"] = "a" * 40
    previous = tmp_path / "f0.json"
    write_rung_evidence(
        previous,
        LadderRung.from_mapping(manifest_mapping("F0")),
        previous_report,
        signing_key=private,
        format_version=2,
    )
    public_key = tmp_path / "signing.pub"
    public_key.write_bytes(private.public_key)
    report = valid_v2_report()
    del report["worker_exit_codes"]
    del report["cost_usd"]
    del report["destruction"]
    module.verify_ladder = lambda deployment: report
    module.stop_workers = lambda workers, revision: [0, 0, 0, 0]
    module.source_revision = lambda: "a" * 40
    module.main(
        [
            str(manifest),
            "--existing-deployment",
            str(deployment),
            "--previous-evidence",
            str(previous),
            "--evidence-public-key",
            str(public_key),
            "--output",
            str(tmp_path / "evidence"),
            "--signing-key",
            str(signing_key),
            "--uncapped",
        ]
    )
    rung, verified = verify_rung_evidence(
        tmp_path / "evidence" / "evidence.json", public_key=private.public_key
    )
    assert rung.rung == "L0"
    assert verified["worker_exit_codes"] == [0, 0, 0, 0]
    assert verified["cost_usd"] == 0
    assert verified["source_revision"] == "a" * 40


def test_ladder_rejects_mismatched_predecessor_revision(tmp_path: Path) -> None:
    from scripts.fortress.run_ladder import verify_predecessor

    from xurcroh.fortress.ladder import LadderEvidenceError

    private = HybridSigningKey.generate()
    report = valid_v2_report()
    report["hardware_validated"] = False
    report["source_revision"] = "a" * 40
    previous = tmp_path / "f0.json"
    write_rung_evidence(
        previous,
        LadderRung.from_mapping(manifest_mapping("F0")),
        report,
        signing_key=private,
        format_version=2,
    )
    with pytest.raises(LadderEvidenceError, match="different source revision"):
        verify_predecessor(previous, "L0", "b" * 40, private.public_key)


def test_ladder_requires_predecessor_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.fortress.run_ladder as module

    from xurcroh.fortress.ladder import LadderEvidenceError

    monkeypatch.setattr(module, "source_revision", lambda: "a" * 40)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_mapping()))
    args = type(
        "Args",
        (),
        {
            "manifest": manifest,
            "model": None,
            "previous_evidence": None,
            "evidence_public_key": None,
        },
    )()
    with pytest.raises(LadderEvidenceError, match="requires --previous-evidence"):
        module.run(args)
