from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
from xurcroh.shards.recipients import StageRecipient


def _verify(args: argparse.Namespace) -> int:
    import json
    import sys

    from xurcroh.receipts import TraceError, describe_trace, verify_trace

    try:
        document = json.loads(args.trace.read_text())
        report = verify_trace(
            document,
            public_key=None if args.public_key is None else bytes.fromhex(args.public_key),
            messages=None if args.messages is None else json.loads(args.messages.read_text()),
            output=None if args.output is None else args.output.read_text(),
        )
    except (TraceError, OSError, ValueError) as error:
        print(f"trace rejected: {error}", file=sys.stderr)
        return 1
    pinned = "verified" if args.public_key is not None else "verified against the embedded key"
    print(f"signature: {pinned}")
    for name in ("prompt", "output"):
        print(f"{name}: {'matches' if report[name] else 'not checked'}")
    print("\n".join(describe_trace(document)))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="xurcroh")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat = subparsers.add_parser("chat")
    chat.add_argument("--model", required=True)
    chat.add_argument("--model-path", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0.0)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--checkpoint", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--stages", type=int, required=True)
    compile_parser.add_argument("--boundary-key-env", default="XURCROH_CLIENT_BOUNDARY_KEY")
    compile_parser.add_argument("--stage-recipient", action="append", required=True)
    compile_parser.add_argument("--seed", type=int)
    compile_parser.add_argument("--compute-dtype", choices=["f64", "f32"], default="f64")
    compile_parser.add_argument("--shard-dtype", choices=["f64", "f32", "f16", "bf16"])
    verify = subparsers.add_parser("verify", help="check a signed task trace")
    verify.add_argument("trace", type=Path)
    verify.add_argument("--public-key", help="hex key of the client expected to have signed it")
    verify.add_argument("--messages", type=Path, help="JSON file with the request messages")
    verify.add_argument("--output", type=Path, help="text file with the generated output")
    args = parser.parse_args()
    if args.command == "verify":
        raise SystemExit(_verify(args))
    dtypes = {
        "f64": torch.float64,
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.command == "chat":
        with Xurcroh(
            cluster={"endpoint": "local://cpu"},
            model_path={args.model: args.model_path},
        ) as client:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(response.choices[0].message.content)
    elif args.command == "compile":
        encoded = os.environ.get(args.boundary_key_env)
        if encoded is None:
            raise ConfigurationError(
                f"client boundary key environment variable {args.boundary_key_env} is not set"
            )
        try:
            boundary_key = bytes.fromhex(encoded)
        except ValueError as error:
            raise ConfigurationError("client boundary key must be hexadecimal") from error
        recipients = []
        for value in args.stage_recipient:
            index_text, separator, public_hex = value.partition("=")
            if not separator:
                raise ConfigurationError("stage recipients require INDEX=PUBLIC_KEY_HEX")
            try:
                index = int(index_text)
                public_key = bytes.fromhex(public_hex)
            except ValueError as error:
                raise ConfigurationError("invalid stage recipient") from error
            recipients.append(StageRecipient(index, public_key, f"stage-{index}"))
        result = XurcrohWeightCompiler(
            CompilationConfig(
                checkpoint=args.checkpoint,
                output_dir=args.output,
                stage_count=args.stages,
                boundary_key=boundary_key,
                recipients=tuple(sorted(recipients, key=lambda recipient: recipient.stage_index)),
                seed=args.seed,
                compute_dtype=dtypes[args.compute_dtype],
                shard_dtype=dtypes[args.shard_dtype] if args.shard_dtype else None,
            )
        ).compile()
        print(result.manifest_path)


if __name__ == "__main__":
    main()
