#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from xurcroh.fortress.checkpoints import checkpoint_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable Fortress checkpoint metadata")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    template = json.loads(args.template.read_text())
    if not isinstance(template, dict):
        raise ValueError("checkpoint template must be an object")
    metadata = checkpoint_metadata(
        args.checkpoint,
        repository=str(template["repository"]),
        revision=str(template["revision"]),
        allow_download=args.allow_download,
        license_name=template.get("license"),
    )
    document = {**template, **metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
