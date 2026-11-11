#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def read_json(path: Path | None, default: object) -> object:
    return json.loads(path.read_text()) if path is not None and path.is_file() else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-report", type=Path)
    parser.add_argument("--hardware", type=Path, action="append", default=[])
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--destruction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sdk = read_json(args.sdk_report, {"status": "not_run"})
    ledger = read_json(args.ledger, {"total_cost_usd": 0.0, "entries": []})
    destruction = read_json(args.destruction, {"all_destroyed": False, "instances": []})
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "sdk_validation": sdk,
        "hardware": [read_json(path, {}) for path in args.hardware],
        "spending": ledger,
        "destruction": destruction,
        "hardware_validated": sdk.get("status") == "passed",
        "production_accepted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Xurcroh GPU validation evidence",
        "",
        f"- Revision: `{revision}`",
        f"- SDK validation: **{sdk.get('status', 'unknown')}**",
        f"- Hardware validated: **{report['hardware_validated']}**",
        "- Production accepted: **False**",
        f"- Recorded spend: **${ledger.get('total_cost_usd', 0.0):.4f}**",
        f"- All tracked instances destroyed: **{destruction.get('all_destroyed', False)}**",
        "",
        "This report aggregates command output; it does not upgrade contract tests or simulations into hardware evidence.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
