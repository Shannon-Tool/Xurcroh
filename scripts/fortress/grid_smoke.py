#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fortress.deploy_ladder import deploy, stop_workers

from xurcroh.fortress.ladder_provider import (
    destroy_ladder_instances,
    materialize_ladder_manifest,
)
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.providers.vast import StateStore


def run(args: argparse.Namespace) -> Path:
    output = args.output
    run_directory = output / "run"
    run_directory.mkdir(parents=True, exist_ok=True)
    store = StateStore(args.state)
    deployment: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "evidence_class": "synthetic-independent-CUDA-host-grid",
        "official_model_gate": False,
        "production_accepted": False,
    }
    try:
        template = json.loads(args.template.read_text())
        provisioned = json.loads(args.provisioned_config.read_text())
        manifest = materialize_ladder_manifest(
            template, provisioned, store, run_directory / "manifest.json"
        )
        deployment_path = deploy(manifest, run_directory / "deployment")
        deployment = json.loads(deployment_path.read_text())
        report.update(verify_ladder(deployment))
    except Exception as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        try:
            if deployment is not None:
                report["worker_exit_codes"] = stop_workers(
                    deployment["workers"], deployment["source_revision"]
                )
        finally:
            if args.state.is_file():
                report["teardown"] = destroy_ladder_instances(store, uncapped=args.uncapped)
                watchdog_pid = store.snapshot().get("watchdog_pid")
                if watchdog_pid:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(watchdog_pid), signal.SIGTERM)
            report["source_revision"] = (
                deployment["source_revision"] if deployment is not None else None
            )
            destination = run_directory / "smoke-report.json"
            destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic Fortress grid without claiming an official model gate"
    )
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--provisioned-config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
