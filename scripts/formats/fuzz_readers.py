#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats import open_checkpoint


def mutate(source: bytes, rng: random.Random) -> bytes:
    payload = bytearray(source)
    operation = rng.choice(("truncate", "flip", "insert", "header"))
    if operation == "truncate" and payload:
        del payload[rng.randrange(len(payload)) :]
    elif operation == "flip" and payload:
        index = rng.randrange(len(payload))
        payload[index] ^= 1 << rng.randrange(8)
    elif operation == "insert":
        payload.extend(rng.randbytes(rng.randrange(1, 32)))
    else:
        payload[:4] = rng.choice((b"GGUF", b"GGU!", b"BAD!", b"GGUF"))
    return bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzz quantized checkpoint readers with bounded inputs"
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100_000:
        raise ValueError("iterations must be between 1 and 100000")
    source = args.fixture.read_bytes()
    rng = random.Random(args.seed)  # noqa: S311 - deterministic test inputs are not cryptographic keys
    accepted = 0
    rejected = 0
    for _ in range(args.iterations):
        candidate = mutate(source, rng)
        args.fixture.write_bytes(candidate)
        try:
            with open_checkpoint(args.fixture) as reader:
                reader.tensor_names()
                accepted += 1
        except (CapabilityError, ModelLoadError, ValueError, KeyError, IndexError, struct_error()):
            rejected += 1
    print(f"accepted={accepted} rejected={rejected}")


def struct_error() -> type[Exception]:
    import struct

    return struct.error


if __name__ == "__main__":
    main()
