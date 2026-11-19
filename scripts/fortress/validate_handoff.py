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
from scripts.fortress.run_ladder import PREDECESSORS

from xurcroh.fortress.ladder import LadderRung

ZERO_REVISION = "0" * 40
ZERO_FINGERPRINT = "0" * 64
PREDECESSOR_OUTPUTS = {
    "L0": "f0-tiny-llama",
    "L1": "l0-gemma-4-e2b",
    "L2": "l1-gemma-4-12b",
    "L3": "l2-gemma-4-26b-a4b",
    "L4": "l3-qwen-3.8-27b",
    "L5": "l4-dense-100-300b-class",
    "L6": "l5-qwen3-235b-a22b",
    "M1": "m0-synthetic-dit",
    "M2": "m1-pixart-sigma-256",
    "M3": "m2-qwen-image",
    "M4": "m3-wan-2.1-t2v-1.3b",
}


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
        predecessor = PREDECESSORS.get(rung.rung)
        previous_option = (
            ""
            if predecessor is None
            else "--previous-evidence /secure/evidence/"
            f"{PREDECESSOR_OUTPUTS[rung.rung]}/evidence.json "
            "--evidence-public-key /secure/evidence-signing.pub "
        )
        commands.append(
            f"python scripts/fortress/run_ladder.py {path}{model_option} "
            "--provisioned-config /secure/provisioned.json "
            "--state /secure/provider-state.json "
            f"--output /secure/evidence/{rung.rung.lower()}-{rung.model} "
            f"{previous_option}"
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
    if not args.paths:
        multi_model = root / "configs/ladders/multi-model-smoke.example.json"
        document = json.loads(multi_model.read_text())
        if not isinstance(document.get("model_path"), dict) or len(document["model_path"]) < 2:
            raise ValueError("multi-model smoke example requires two models")
        print(
            "python scripts/fortress/multi_model_smoke.py "
            f"--config {multi_model} --output /secure/evidence/multi-model-smoke.json"
        )


if __name__ == "__main__":
    main()
