from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

from xurcroh.fortress.relay import ObliviousRelay
from xurcroh.fortress.transport import MutualTLSIdentity
from xurcroh.memory_host import protect_process


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a Fortress onion relay that forwards only to allowlisted next hops"
    )
    parser.add_argument("--onion-key", required=True, type=Path, help="raw 32-byte X25519 key")
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--ca", required=True, type=Path)
    parser.add_argument("--upstream-certificate", required=True, type=Path)
    parser.add_argument("--upstream-private-key", required=True, type=Path)
    parser.add_argument("--upstream-server-name", default="localhost")
    parser.add_argument(
        "--allow",
        action="append",
        required=True,
        metavar="quic://HOST:PORT",
        help="next hop this relay may forward to; repeat for each relay or worker",
    )
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9445)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    protect_process()
    onion_key = args.onion_key.read_bytes()
    if len(onion_key) != 32:
        raise ValueError("relay onion key must be exactly 32 bytes")
    ca = args.ca.read_bytes()
    relay = ObliviousRelay(
        onion_key,
        certificates=(args.certificate.read_bytes(), ca),
        private_key=args.private_key.read_bytes(),
        client_ca=ca,
        upstream_identity=MutualTLSIdentity(
            certificates=(args.upstream_certificate.read_bytes(), ca),
            private_key=args.upstream_private_key.read_bytes(),
            pinned_ca=ca,
            server_name=args.upstream_server_name,
        ),
        allowed_workers=set(args.allow),
        address=f"{args.host}:{args.port}",
    )
    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    relay.start()
    if args.ready_file is not None:
        temporary = args.ready_file.with_suffix(".tmp")
        temporary.write_text(f"{relay.address}\n")
        temporary.replace(args.ready_file)
    try:
        stopped.wait()
    finally:
        relay.close()


if __name__ == "__main__":
    main()
