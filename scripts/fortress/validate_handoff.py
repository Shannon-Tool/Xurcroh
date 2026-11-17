#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from xurcroh.fortress.ladder import LadderRung

ZERO_REVISION = "0" * 40
ZERO_FINGERPRINT = "0" * 64


def template_workers() -> list[dict[str, Any]]:
    diversity = (
        ("provider-a", "us", "amd"),
        ("provider-b", "de", "intel"),
        ("provider-a", "de", "intel"),
        ("provider-b", "us", "amd"),
    )
    workers = []
    for index, (provider, country, cpu) in enumerate(diversity):
        activation, weight = divmod(index, 2)
        workers.append(
            {
                "name": f"role-{activation}-{weight}",
                "endpoint": f"provision://role-{index}",
                "machine_id": index + 1,
                "provider_id": provider,
                "country": country,
                "cpu_vendor": cpu,
                "activation_index": activation,
                "weight_index": weight,
                "recipient_public": bytes([index + 1] * 32).hex(),
            }
        )
    return workers


def documents(path: Path) -> list[dict[str, Any]]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain an object")
    rungs = document.get("rungs", [document])
    if not isinstance(rungs, list) or not all(isinstance(rung, dict) for rung in rungs):
        raise ValueError(f"{path} has an invalid rung list")
    return rungs


def validate(path: Path) -> list[str]:
    commands = []
    for document in documents(path):
        provisional = "workers" not in document
        if provisional:
            document = {**document, "workers": template_workers()}
        rung = LadderRung.from_mapping(document, allow_unverified_hardware=True)
        if document.get("resolution_required") is True:
            if rung.checkpoint_fingerprint != ZERO_FINGERPRINT:
                raise ValueError(f"{rung.rung} unresolved artifact must use a zero fingerprint")
        elif rung.revision == ZERO_REVISION or rung.checkpoint_fingerprint == ZERO_FINGERPRINT:
            raise ValueError(f"{rung.rung} requires immutable checkpoint provenance")
        model_option = f" --model {rung.model}" if len(documents(path)) > 1 else ""
        commands.append(
            f"python scripts/fortress/run_ladder.py {path}{model_option} "
            "--provisioned-config /secure/provisioned.json "
            "--state /secure/provider-state.json "
            f"--output /secure/evidence/{rung.rung.lower()}-{rung.model} "
            "--signing-key /secure/evidence-signing.key"
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate public GPU handoff manifests offline")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = args.paths or sorted((root / "configs/ladders").glob("*.yaml"))
    if not paths:
        raise SystemExit("no ladder manifests found")
    for path in paths:
        for command in validate(path):
            print(command)


if __name__ == "__main__":
    main()
