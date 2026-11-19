#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Xurcroh import Xurcroh
from xurcroh.fortress.ladder_verifier import _verify_text_lifecycle


def run(config_path: Path, output: Path) -> Path:
    configuration = json.loads(config_path.read_text())
    if not isinstance(configuration, dict):
        raise ValueError("multi-model configuration must be an object")
    models = configuration.get("model_path")
    if not isinstance(models, dict) or len(models) < 2:
        raise ValueError("multi-model smoke requires at least two registered models")
    if "cluster" not in configuration:
        raise ValueError("multi-model configuration requires a cluster")
    names = list(models)
    messages = [{"role": "user", "content": "Reply with a short technical sentence."}]
    with Xurcroh(**configuration) as client:
        responses = {
            name: client.chat.completions.create(
                model=name,
                messages=messages,
                temperature=0,
                max_tokens=1,
            )
            for name in names
        }
        lifecycle = _verify_text_lifecycle(client, names[0])
        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            concurrent = list(
                executor.map(
                    lambda name: client.chat.completions.create(
                        model=name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1,
                    ),
                    names,
                )
            )
        metrics: dict[str, Any] = client.metrics()
    report = {
        "schema_version": 1,
        "production_accepted": False,
        "models": {name: response.model_dump(mode="json") for name, response in responses.items()},
        "lifecycle": lifecycle,
        "concurrent_requests": len(concurrent),
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute real multi-model Fortress SDK smoke checks"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
