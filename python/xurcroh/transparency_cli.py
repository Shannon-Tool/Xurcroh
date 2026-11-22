from __future__ import annotations

import argparse
import os
from pathlib import Path

from xurcroh.signatures import load_signing_key
from xurcroh.transparency_service import TransparencyService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the Xurcroh public transparency log")
    parser.add_argument("--storage", required=True, type=Path)
    parser.add_argument("--signing-key", required=True, type=Path)
    parser.add_argument("--write-token-env", default="XURCROH_TRANSPARENCY_WRITE_TOKEN")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    args = parser.parse_args(argv)
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        parser.error("--ssl-certfile and --ssl-keyfile must be supplied together")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        write_token = os.environ[args.write_token_env]
    except KeyError as error:
        raise ValueError(
            f"write token environment variable {args.write_token_env} is unset"
        ) from error
    signing_key = load_signing_key(args.signing_key)
    service = TransparencyService(
        args.storage,
        signing_key=signing_key,
        write_token=write_token,
    )
    import uvicorn

    options = {
        "host": args.host,
        "port": args.port,
        "access_log": False,
    }
    if args.ssl_certfile is not None:
        options["ssl_certfile"] = str(args.ssl_certfile)
        options["ssl_keyfile"] = str(args.ssl_keyfile)
    uvicorn.run(service.app, **options)


if __name__ == "__main__":
    main()
