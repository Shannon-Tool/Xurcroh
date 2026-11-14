#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from xurcroh.release import verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a signed Xurcroh release")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--public-key", required=True, help="Ed25519 public key in hex")
    args = parser.parse_args()
    verify_release(
        args.manifest,
        public_key=Ed25519PublicKey.from_public_bytes(bytes.fromhex(args.public_key)),
    )
    print(f"Verified release manifest {args.manifest}")


if __name__ == "__main__":
    main()
