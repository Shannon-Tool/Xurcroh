from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from xurcroh.fortress.ladder import (
    LadderEvidenceError,
    LadderRung,
    run_ladder_rung,
    validate_rung_evidence,
    verify_rung_evidence,
    write_rung_evidence,
)


def manifest_mapping(rung: str = "L0") -> dict:
    workers = [
        {
            "name": f"role-{activation}-{weight}",
            "endpoint": f"192.0.2.{activation * 2 + weight + 1}:9444",
            "machine_id": activation * 2 + weight + 1,
            "provider_id": f"provider-{activation * 2 + weight}",
            "country": f"country-{activation * 2 + weight}",
            "cpu_vendor": "amd" if (activation + weight) % 2 else "intel",
            "activation_index": activation,
            "weight_index": weight,
            "recipient_public": f"{activation * 2 + weight:02x}" * 32,
        }
        for activation in range(2)
        for weight in range(2)
    ]
    return {
        "rung": rung,
        "model": "gemma-e2b",
        "modality": "text",
        "repository": "google/gemma-4-e2b-it",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "checkpoint_fingerprint": "b" * 64,
        "share_count": 2,
        "workers": workers,
    }


def evidence_mapping(rung: str = "L0") -> dict:
    return {
        "hardware_validated": rung not in {"F0", "M0"},
        "production_accepted": False,
        "quantized_reference_max_abs_error": 0.0,
        "original_model_max_abs_error": 0.01,
        "attack_results": {"single_share_reconstructs": False},
        "freivalds_failures": 0,
        "canary_tripped": True,
        "worker_exit_codes": [0, 0, 0, 0],
        "bytes_sent": 100,
        "bytes_received": 100,
        "inference_seconds": 1.0,
        "cost_usd": 2.0,
        "destruction": {"all_destroyed": True, "remaining_instance_ids": []},
    }


def test_ladder_manifest_builds_complete_diverse_grid() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    assert rung.rung == "L0"
    assert len(rung.workers) == 4
    assert {(worker.activation_index, worker.weight_index) for worker in rung.workers} == {
        (activation, weight) for activation in range(2) for weight in range(2)
    }


def test_ladder_manifest_rejects_duplicate_machines() -> None:
    document = manifest_mapping()
    document["workers"][1]["machine_id"] = document["workers"][0]["machine_id"]
    with pytest.raises(LadderEvidenceError, match="machine"):
        LadderRung.from_mapping(document)


def test_hardware_evidence_requires_exact_results_and_teardown() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    evidence = evidence_mapping()
    validate_rung_evidence(rung, evidence)
    evidence["destruction"]["all_destroyed"] = False
    with pytest.raises(LadderEvidenceError, match="teardown"):
        validate_rung_evidence(rung, evidence)


def test_hardware_evidence_rejects_numerical_mismatch() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    evidence = evidence_mapping()
    evidence["quantized_reference_max_abs_error"] = 0.1
    with pytest.raises(LadderEvidenceError, match="quantized"):
        validate_rung_evidence(rung, evidence)


def test_local_rungs_do_not_claim_hardware_validation() -> None:
    local = LadderRung.from_mapping(manifest_mapping("F0"))
    validate_rung_evidence(local, evidence_mapping("F0"))
    hardware = LadderRung.from_mapping(manifest_mapping("L0"))
    evidence = evidence_mapping("L0")
    evidence["hardware_validated"] = False
    with pytest.raises(LadderEvidenceError, match="hardware"):
        validate_rung_evidence(hardware, evidence)


def test_signed_ladder_evidence_verifies_and_detects_tampering(tmp_path: Path) -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    path = tmp_path / "l0.json"
    rung = LadderRung.from_mapping(manifest_mapping())
    write_rung_evidence(path, rung, evidence_mapping(), signing_key=private)
    verified_rung, report = verify_rung_evidence(path, public_key=public)
    assert verified_rung == rung
    assert report["freivalds_failures"] == 0
    document = json.loads(path.read_text())
    document["report"]["cost_usd"] = 0
    path.write_text(json.dumps(document))
    with pytest.raises(LadderEvidenceError, match="signature"):
        verify_rung_evidence(path, public_key=public)


def test_ladder_execution_tears_down_before_acceptance() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    events: list[str] = []

    def execute(_rung: LadderRung) -> dict:
        events.append("execute")
        report = evidence_mapping()
        del report["destruction"]
        return report

    def teardown() -> dict:
        events.append("teardown")
        return {"all_destroyed": True, "remaining_instance_ids": []}

    report = run_ladder_rung(rung, execute=execute, teardown=teardown)
    assert events == ["execute", "teardown"]
    assert report["destruction"]["all_destroyed"] is True


def test_ladder_execution_failure_still_tears_down() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    destroyed: list[str] = []

    def execute(_rung: LadderRung) -> dict:
        raise RuntimeError("inference failed")

    def teardown() -> dict:
        destroyed.append("roles")
        return {"all_destroyed": True, "remaining_instance_ids": []}

    with pytest.raises(RuntimeError, match="inference failed"):
        run_ladder_rung(rung, execute=execute, teardown=teardown)
    assert destroyed == ["roles"]


def test_ladder_execution_rejects_failed_teardown() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())

    def execute(_rung: LadderRung) -> dict:
        report = evidence_mapping()
        del report["destruction"]
        return report

    with pytest.raises(LadderEvidenceError, match="teardown"):
        run_ladder_rung(
            rung,
            execute=execute,
            teardown=lambda: {"all_destroyed": False, "remaining_instance_ids": [1]},
        )


def test_ladder_teardown_replaces_preliminary_cost() -> None:
    rung = LadderRung.from_mapping(manifest_mapping())

    def execute(_rung: LadderRung) -> dict:
        report = evidence_mapping()
        del report["destruction"]
        return report

    report = run_ladder_rung(
        rung,
        execute=execute,
        teardown=lambda: {
            "destruction": {"all_destroyed": True, "remaining_instance_ids": []},
            "cost_usd": 3.5,
        },
    )
    assert report["cost_usd"] == 3.5
