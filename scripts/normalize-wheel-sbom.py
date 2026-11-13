#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from xurcroh.release import normalize_wheel_sbom


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize build paths in native wheel SBOMs")
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        normalize_wheel_sbom(wheel, project_root=args.project_root)


if __name__ == "__main__":
    main()
