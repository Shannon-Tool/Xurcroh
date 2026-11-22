#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xurcroh.release import verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a signed Xurcroh release")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--public-key", required=True, help="hybrid signing public key in hex")
    parser.add_argument("--log-url", help="public log that must hold this release's digest")
    parser.add_argument("--log-key", help="hex key of that log")
    args = parser.parse_args()
    if (args.log_url is None) != (args.log_key is None):
        parser.error("--log-url and --log-key must be given together")
    verify_release(args.manifest, public_key=bytes.fromhex(args.public_key))
    print(f"Verified release manifest {args.manifest}")
    if args.log_url is not None:
        import hashlib

        from xurcroh.public_log import LogClient

        digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        proof = LogClient(args.log_url, bytes.fromhex(args.log_key)).find("release", digest)
        if proof is None:
            raise SystemExit(f"release {digest} is not in the public log")
        print(f"Release {digest} is in the public log at index {proof['leaf_index']}")


if __name__ == "__main__":
    main()
