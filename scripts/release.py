#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from xurcroh.build import wheel_build_digest
from xurcroh.public_log import LogClient
from xurcroh.release import generate_sbom, sign_release
from xurcroh.signatures import load_signing_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBOM and sign a Fortress release")
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--log-url", help="public log to publish the release digest to")
    parser.add_argument("--log-key", help="hex key of that log")
    parser.add_argument("--log-token-env", default="XURCROH_LOG_WRITE_TOKEN")
    args = parser.parse_args()
    if (args.log_url is None) != (args.log_key is None):
        parser.error("--log-url and --log-key must be given together")
    private = load_signing_key(args.signing_key)
    generate_sbom(args.project, args.sbom)
    manifest = sign_release(args.artifacts, sbom_path=args.sbom, signing_key=private)
    print(f"Signed {manifest}; public key: {private.public_key.hex()}")
    if args.log_url is not None:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        token = os.environ.get(args.log_token_env)
        log = LogClient(args.log_url, bytes.fromhex(args.log_key), write_token=token)
        proof = log.publish("release", digest)
        print(f"Logged release {digest} at index {proof['leaf_index']}")
        wheels = sorted(args.artifacts.glob("*.whl"))
        if wheels:
            # What an installation of these wheels reports, so clients can require it.
            build = wheel_build_digest(wheels)
            proof = log.publish("build", build)
            print(f"Logged client build {build} at index {proof['leaf_index']}")


if __name__ == "__main__":
    main()
