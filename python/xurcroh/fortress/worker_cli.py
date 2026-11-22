from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path
from urllib.parse import quote, unquote

from xurcroh.fortress.worker import FortressRoleWorker
from xurcroh.kem import PRIVATE_KEY_BYTES
from xurcroh.memory_host import protect_process


def artifact_file_name(operation: str) -> str:
    if not operation:
        raise ValueError("operation name cannot be empty")
    return quote(operation, safe="")


def load_role_artifacts(directory: str | Path) -> dict[str, bytes]:
    root = Path(directory)
    artifacts = {
        unquote(path.name.removesuffix(".json")): path.read_bytes()
        for path in sorted(root.glob("*.json"))
        if path.is_file()
    }
    if not artifacts or any(not name for name in artifacts):
        raise ValueError("Fortress role worker requires at least one named artifact")
    return artifacts


def record_exit_status(path: Path, code: int) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"{code}\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an isolated Fortress role worker")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--recipient-key", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--ca", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9444)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-unattested",
        action="store_true",
        help="explicit opt-in for non-TEE execution without hardware attestation",
    )
    parser.add_argument(
        "--lock-memory",
        action="store_true",
        help="lock all process memory so secrets cannot be swapped to disk",
    )
    parser.add_argument(
        "--onion-key",
        type=Path,
        help="hybrid onion key created with python -m xurcroh.kem; when set, only "
        "onion-wrapped relay traffic is accepted",
    )
    parser.add_argument(
        "--identity-key",
        type=Path,
        help="hybrid signing key; when set, the worker signs session checkpoints on request",
    )
    parser.add_argument("--exit-status-file", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args(argv)
    exit_code = 0
    try:
        protect_process(lock_memory=args.lock_memory)
        onion_key = None if args.onion_key is None else args.onion_key.read_bytes()
        if onion_key is not None and len(onion_key) != PRIVATE_KEY_BYTES:
            raise ValueError(f"worker onion key must be exactly {PRIVATE_KEY_BYTES} bytes")
        worker = FortressRoleWorker(
            load_role_artifacts(args.artifact_dir),
            args.recipient_key.read_bytes(),
            certificates=(args.certificate.read_bytes(), args.ca.read_bytes()),
            private_key=args.private_key.read_bytes(),
            client_ca=args.ca.read_bytes(),
            address=f"{args.host}:{args.port}",
            device=args.device,
            allow_unattested=args.allow_unattested,
            onion_private_key=onion_key,
            identity_key=None if args.identity_key is None else args.identity_key.read_bytes(),
        )
        stopped = threading.Event()

        def stop(_signum: int, _frame: object) -> None:
            stopped.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        worker.start()
        if args.pid_file is not None:
            args.pid_file.write_text(f"{os.getpid()}\n")
        if args.ready_file is not None:
            args.ready_file.write_text(f"{worker.address}\n")
        try:
            stopped.wait()
        finally:
            worker.close()
    except BaseException:
        exit_code = 1
        raise
    finally:
        if args.exit_status_file is not None:
            record_exit_status(args.exit_status_file, exit_code)


if __name__ == "__main__":
    main()
