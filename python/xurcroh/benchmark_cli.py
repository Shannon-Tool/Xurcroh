from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import yaml

from Xurcroh import Xurcroh
from xurcroh.benchmarking import (
    benchmark_chat,
    benchmark_fwht,
    benchmark_image,
    benchmark_memory,
    benchmark_transport,
    benchmark_video,
    write_report,
)
from xurcroh.errors import ConfigurationError


def load_runtime_config(path: str | Path) -> Xurcroh:
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, Mapping) or "model_path" not in values:
        raise ConfigurationError("benchmark configuration requires a model_path mapping")
    allowed = {
        "cluster",
        "model_path",
        "pipeline_parallel",
        "tensor_parallel",
        "expert_parallel",
        "security",
        "provision_workers",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(f"unknown benchmark configuration fields: {sorted(unknown)}")
    return Xurcroh(**dict(values))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xurcroh-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    def output(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output", type=Path, required=True)

    def model(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--prompt", required=True)
        command.add_argument("--steps", type=int, default=30)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--timeout", type=float)
        command.add_argument("--reference", type=Path)
        output(command)

    chat = commands.add_parser("chat")
    chat.add_argument("--config", type=Path, required=True)
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--iterations", type=int, default=5)
    chat.add_argument("--max-tokens", type=int, default=32)
    chat.add_argument("--temperature", type=float, default=0)
    chat.add_argument("--timeout", type=float)
    output(chat)

    image = commands.add_parser("image")
    model(image)
    image.add_argument("--size", default="1024x1024")

    video = commands.add_parser("video")
    model(video)
    video.add_argument("--size", default="832x480")
    video.add_argument("--frames", type=int, default=81)
    video.add_argument("--fps", type=int, default=24)

    transform = commands.add_parser("fwht")
    transform.add_argument("--dimension", type=int, default=4096)
    transform.add_argument("--iterations", type=int, default=20)
    transform.add_argument("--seed", type=int, default=0)
    output(transform)

    transport = commands.add_parser("transport")
    transport.add_argument("--payload-bytes", type=int, default=65536)
    transport.add_argument("--iterations", type=int, default=100)
    output(transport)

    memory = commands.add_parser("memory")
    memory.add_argument("--dimension", type=int, default=4096)
    memory.add_argument("--rows", type=int, default=128)
    memory.add_argument("--iterations", type=int, default=20)
    memory.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    output(memory)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fwht":
        report = benchmark_fwht(
            dimension=args.dimension, iterations=args.iterations, seed=args.seed
        )
    elif args.command == "transport":
        report = benchmark_transport(payload_bytes=args.payload_bytes, iterations=args.iterations)
    elif args.command == "memory":
        report = benchmark_memory(
            dimension=args.dimension,
            rows=args.rows,
            iterations=args.iterations,
            device=args.device,
        )
    else:
        with load_runtime_config(args.config) as client:
            if args.command == "chat":
                report = benchmark_chat(
                    client,
                    model=args.model,
                    messages=[{"role": "user", "content": args.prompt}],
                    iterations=args.iterations,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            elif args.command == "image":
                report = benchmark_image(
                    client,
                    model=args.model,
                    prompt=args.prompt,
                    size=args.size,
                    steps=args.steps,
                    seed=args.seed,
                    reference=args.reference,
                    timeout=args.timeout,
                )
            elif args.command == "video":
                report = benchmark_video(
                    client,
                    model=args.model,
                    prompt=args.prompt,
                    size=args.size,
                    frames=args.frames,
                    fps=args.fps,
                    steps=args.steps,
                    seed=args.seed,
                    reference=args.reference,
                    timeout=args.timeout,
                )
            else:
                raise ConfigurationError(f"unsupported benchmark command: {args.command}")
    write_report(report, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
