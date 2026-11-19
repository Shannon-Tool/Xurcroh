#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from scripts.fortress.deploy_ladder import (
    collect_role_resources,
    deploy,
    source_revision,
    stop_workers,
)

from xurcroh.fortress.ladder import (
    LadderEvidenceError,
    LadderRung,
    run_ladder_rung,
    verify_rung_evidence,
    write_rung_evidence,
)
from xurcroh.fortress.ladder_provider import (
    destroy_ladder_instances,
    materialize_ladder_manifest,
)
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.providers.vast import StateStore


def load_rung_manifest(path: Path, model: str | None) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("ladder template must be an object")
    if "rungs" not in document:
        return document
    rungs = document["rungs"]
    if not isinstance(rungs, list):
        raise ValueError("ladder rungs must be a list")
    if model is None:
        raise ValueError("multi-model ladder manifests require --model")
    selected = [rung for rung in rungs if isinstance(rung, dict) and rung.get("model") == model]
    if len(selected) != 1:
        raise ValueError(f"manifest does not contain exactly one model named {model!r}")
    return {**{key: value for key, value in document.items() if key != "rungs"}, **selected[0]}


PREDECESSORS = {
    "L0": "F0",
    "L1": "L0",
    "L2": "L1",
    "L3": "L2",
    "L4": "L3",
    "L5": "L4",
    "L6": "L5",
    "M1": "M0",
    "M2": "M1",
    "M3": "M2",
    "M4": "M3",
}


def verify_predecessor(path: Path, rung: str, revision: str, public_key: Ed25519PublicKey) -> None:
    expected = PREDECESSORS.get(rung)
    if expected is None:
        raise LadderEvidenceError(f"{rung} is a root rung and has no predecessor")
    previous_rung, report = verify_rung_evidence(path, public_key=public_key)
    if previous_rung.rung != expected:
        raise LadderEvidenceError(f"{rung} requires passing {expected} evidence")
    if report.get("source_revision") != revision:
        raise LadderEvidenceError("predecessor evidence uses a different source revision")


def run(args: argparse.Namespace) -> Path:
    template = load_rung_manifest(args.manifest, args.model)
    revision = source_revision()
    expected = PREDECESSORS.get(template.get("rung"))
    if expected is not None:
        if args.previous_evidence is None or args.evidence_public_key is None:
            raise LadderEvidenceError(
                f"{template['rung']} requires --previous-evidence and --evidence-public-key"
            )
        verify_predecessor(
            args.previous_evidence,
            template["rung"],
            revision,
            Ed25519PublicKey.from_public_bytes(args.evidence_public_key.read_bytes()),
        )
    elif args.previous_evidence is not None:
        raise LadderEvidenceError("root rungs do not accept predecessor evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest-template.yaml"
    manifest.write_text(yaml.safe_dump(template, sort_keys=True))
    store = StateStore(args.state)
    provisioned = None
    if args.provisioned_config is not None:
        provisioned = json.loads(args.provisioned_config.read_text())
        manifest = materialize_ladder_manifest(
            template, provisioned, store, args.output / "manifest.json"
        )
    rung = LadderRung.from_mapping(
        yaml.safe_load(manifest.read_text()), allow_unverified_hardware=True
    )

    def execute(_rung: LadderRung) -> dict[str, Any]:
        nonlocal rung
        deployment_path = args.existing_deployment or deploy(
            manifest, args.output, skip_bootstrap=args.skip_bootstrap
        )
        deployment = json.loads(Path(deployment_path).read_text())
        if "rung" in deployment:
            rung = LadderRung.from_mapping(deployment["rung"])
        try:
            report = verify_ladder(deployment)
            report["role_resources"] = collect_role_resources(
                deployment["workers"], deployment["source_revision"]
            )
        finally:
            exit_codes = stop_workers(deployment["workers"], deployment["source_revision"])
        report["worker_exit_codes"] = exit_codes
        report["source_revision"] = deployment["source_revision"]
        report["cost_usd"] = 0.0
        return report

    def teardown() -> dict[str, Any]:
        if args.provisioned_config is None:
            return {
                "destruction": {"all_destroyed": True, "remaining_instance_ids": []},
                "cost_usd": 0.0,
            }
        return destroy_ladder_instances(store, uncapped=args.uncapped, provisioned=provisioned)

    report = run_ladder_rung(rung, execute=execute, teardown=teardown)
    signing_key = Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
    destination = args.output / "evidence.json"
    write_rung_evidence(destination, rung, report, signing_key=signing_key, format_version=2)
    print(destination)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Execute and attest one Fortress ladder rung")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--provisioned-config", type=Path)
    parser.add_argument("--existing-deployment", type=Path)
    parser.add_argument("--previous-evidence", type=Path)
    parser.add_argument("--evidence-public-key", type=Path)
    parser.add_argument("--state", type=Path, default=Path("artifacts/vast/state.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    if args.existing_deployment is not None and args.provisioned_config is not None:
        parser.error("--existing-deployment and --provisioned-config are mutually exclusive")
    run(args)


if __name__ == "__main__":
    main()
