from __future__ import annotations

import argparse
import os
from pathlib import Path

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler


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
    compile_parser.add_argument("--master-key-env", default="XURCROH_MASTER_KEY")
    compile_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
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
        encoded = os.environ.get(args.master_key_env)
        if encoded is None:
            raise ConfigurationError(
                f"master key environment variable {args.master_key_env} is not set"
            )
        try:
            master_key = bytes.fromhex(encoded)
        except ValueError as error:
            raise ConfigurationError("master key must be hexadecimal") from error
        result = XurcrohWeightCompiler(
            CompilationConfig(
                checkpoint=args.checkpoint,
                output_dir=args.output,
                stage_count=args.stages,
                master_key=master_key,
                seed=args.seed,
            )
        ).compile()
        print(result.manifest_path)


if __name__ == "__main__":
    main()
