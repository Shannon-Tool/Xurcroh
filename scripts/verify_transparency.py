#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from xurcroh.transparency import verify_transparency_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a signed Fortress transparency log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--public-key", required=True, help="Ed25519 public key in hex")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="IMAGE=MEASUREMENT",
        help="required worker image measurement; repeat for every required image",
    )
    args = parser.parse_args()
    expected = {}
    for value in args.expect:
        image, separator, measurement = value.partition("=")
        if not separator or not image or not measurement:
            raise ValueError("--expect requires IMAGE=MEASUREMENT")
        expected[image] = measurement
    verify_transparency_log(
        json.loads(args.log.read_text()),
        public_key=bytes.fromhex(args.public_key),
        expected_measurements=expected,
    )
    print(f"Verified {args.log} with {len(expected)} required measurements")


if __name__ == "__main__":
    main()
