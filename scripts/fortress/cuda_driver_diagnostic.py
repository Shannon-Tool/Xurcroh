#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from xurcroh.fortress.cuda_driver import (
    collect_driver_diagnostics,
    is_compat_library,
    kernel_driver_version,
)

__all__ = [
    "collect_driver_diagnostics",
    "is_compat_library",
    "kernel_driver_version",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CUDA driver and loader compatibility")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect_driver_diagnostics()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
