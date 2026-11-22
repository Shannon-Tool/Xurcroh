#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from xurcroh.release import generate_sbom, sign_release
from xurcroh.signatures import load_signing_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBOM and sign a Fortress release")
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    args = parser.parse_args()
    private = load_signing_key(args.signing_key)
    generate_sbom(args.project, args.sbom)
    manifest = sign_release(args.artifacts, sbom_path=args.sbom, signing_key=private)
    print(f"Signed {manifest}; public key: {private.public_key.hex()}")


if __name__ == "__main__":
    main()
