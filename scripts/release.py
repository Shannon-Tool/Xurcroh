#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.release import generate_sbom, sign_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBOM and sign a Fortress release")
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    args = parser.parse_args()
    private_bytes = args.signing_key.read_bytes()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.from_private_bytes(private_bytes)
    generate_sbom(args.project, args.sbom)
    manifest = sign_release(args.artifacts, sbom_path=args.sbom, signing_key=private)
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    print(f"Signed {manifest}; public key: {public}")


if __name__ == "__main__":
    main()
