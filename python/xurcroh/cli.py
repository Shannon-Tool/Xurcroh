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

    from xurcroh.errors import XurcrohError
    from xurcroh.receipts import TraceError, describe_trace, verify_trace, verify_trace_proof

    try:
        document = json.loads(args.trace.read_text())
        report = verify_trace(
            document,
            public_key=None if args.public_key is None else bytes.fromhex(args.public_key),
            messages=None if args.messages is None else json.loads(args.messages.read_text()),
            media_input=None
            if args.media_input is None
            else json.loads(args.media_input.read_text()),
            output=None if args.output is None else args.output.read_text(),
        )
        logged = None
        live = None
        build_logged: bool | None = None
        witnessed: list[str] = []
        if args.proof is None and (args.log_url or args.witness_key):
            raise TraceError("checking the public log or its witnesses requires --proof")
        if args.proof is not None:
            if args.log_key is None:
                raise TraceError("checking a log proof requires --log-key")
            from xurcroh.public_log import LogClient, witnesses_of
            from xurcroh.signatures import key_fingerprint

            log_key = bytes.fromhex(args.log_key)
            proof = json.loads(args.proof.read_text())
            logged = verify_trace_proof(document, proof, log_key=log_key)
            heads = [proof["tree_head"]]
            if args.log_url is not None:
                # Shows the log, as served now, still contains the history this proof is from.
                heads.append(LogClient(args.log_url, log_key).extends(proof["tree_head"]))
                live = int(heads[-1]["tree_size"])
                build = (document.get("client") or {}).get("build_sha256")
                if isinstance(build, str):
                    build_logged = LogClient(args.log_url, log_key).find("build", build) is not None
            trusted = [bytes.fromhex(key) for key in args.witness_key]
            seen = {name for head in heads for name in witnesses_of(head, trusted)}
            missing = [key_fingerprint(key) for key in trusted if key_fingerprint(key) not in seen]
            if missing:
                raise TraceError(
                    "no countersignature from required witness " + ", ".join(sorted(missing))
                )
            witnessed = sorted(seen)
    except (TraceError, XurcrohError, OSError, ValueError) as error:
        print(f"trace rejected: {error}", file=sys.stderr)
        return 1
    pinned = "verified" if args.public_key is not None else "verified against the embedded key"
    print(f"signature: {pinned}")
    for name in ("prompt", "output"):
        print(f"{name}: {'matches' if report[name] else 'not checked'}")
    print(f"log: {'not checked' if logged is None else f'included at tree size {logged}'}")
    if live is not None:
        print(f"public log: still holds that history, now at tree size {live}")
    if build_logged is not None:
        # The digest is the client's own report; this shows only that such a build was published.
        state = "is a published build" if build_logged else "is NOT in the log"
        print(f"client build: the build this trace names {state}")
    if witnessed:
        print("witnesses: " + ", ".join(witnessed))
    signed = report["worker_checkpoints"]
    print(
        "worker checkpoints: "
        + ("none; workers signed nothing" if signed is None else f"{signed} verified")
    )
    if document.get("mode") == "hosted":
        print(
            "note: a hosted gateway produced this trace. The checks above show it was not "
            "changed afterwards; what it says about the model and settings is the "
            "operator's statement, not something you ran."
        )
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
    verify.add_argument(
        "--media-input", type=Path, help="JSON file with an image or video prompt, image and mask"
    )
    verify.add_argument("--output", type=Path, help="text file with the generated output")
    verify.add_argument("--proof", type=Path, help="JSON inclusion proof from tasks.proof")
    verify.add_argument("--log-key", help="hex key of the log expected to hold the trace")
    verify.add_argument(
        "--log-url", help="public log to ask whether it still holds the proof's history"
    )
    verify.add_argument(
        "--witness-key",
        action="append",
        default=[],
        help="hex key of a witness that must have countersigned; repeat for each one",
    )
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
