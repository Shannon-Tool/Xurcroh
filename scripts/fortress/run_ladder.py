#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.fortress.deploy_ladder import deploy, stop_workers

from xurcroh.fortress.ladder import LadderRung, run_ladder_rung, write_rung_evidence
from xurcroh.fortress.ladder_provider import (
    destroy_ladder_instances,
    materialize_ladder_manifest,
)
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.providers.vast import StateStore


def run(args: argparse.Namespace) -> Path:
    manifest = args.manifest
    store = StateStore(args.state)
    if args.provisioned_config is not None:
        template = yaml.safe_load(args.manifest.read_text())
        if not isinstance(template, dict):
            raise ValueError("ladder template must be an object")
        provisioned = json.loads(args.provisioned_config.read_text())
        manifest = materialize_ladder_manifest(
            template, provisioned, store, args.output / "manifest.json"
        )
    rung = LadderRung.from_mapping(yaml.safe_load(manifest.read_text()))

    def execute(_rung: LadderRung) -> dict[str, Any]:
        deployment_path = args.existing_deployment or deploy(
            manifest, args.output, skip_bootstrap=args.skip_bootstrap
        )
        deployment = json.loads(Path(deployment_path).read_text())
        try:
            report = verify_ladder(deployment)
        finally:
            exit_codes = stop_workers(deployment["workers"], deployment["source_revision"])
        report["worker_exit_codes"] = exit_codes
        report["cost_usd"] = 0.0
        return report

    def teardown() -> dict[str, Any]:
        if args.provisioned_config is None:
            return {
                "destruction": {"all_destroyed": True, "remaining_instance_ids": []},
                "cost_usd": 0.0,
            }
        return destroy_ladder_instances(store, uncapped=args.uncapped)

    report = run_ladder_rung(rung, execute=execute, teardown=teardown)
    signing_key = Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
    destination = args.output / "evidence.json"
    write_rung_evidence(destination, rung, report, signing_key=signing_key)
    print(destination)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Execute and attest one Fortress ladder rung")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--provisioned-config", type=Path)
    parser.add_argument("--existing-deployment", type=Path)
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
