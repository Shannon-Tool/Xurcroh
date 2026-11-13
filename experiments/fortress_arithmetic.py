#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from xurcroh.fortress.arithmetic import (
    FIELD_MODULUS,
    PRIME_MODULUS,
    ShareSecurity,
    combine_shares,
    freivalds_challenge,
    seeded_split_secret,
    split_secret,
)


def benchmark(iterations: int = 5) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    generator = torch.Generator().manual_seed(20261112)
    secret = torch.randint(-8, 8, (16, 16), generator=generator, dtype=torch.int64)
    full_storage = 2 * secret.numel() * 4
    seed_storage = 32 + secret.numel() * 4

    started = time.perf_counter()
    for _ in range(iterations):
        full = split_secret(secret, count=2)
    full_ms = (time.perf_counter() - started) * 1000 / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        seeded = seeded_split_secret(secret, seed=bytes(range(32)), count=2)
    seeded_ms = (time.perf_counter() - started) * 1000 / iterations
    assert torch.equal(combine_shares(full.shares), secret.remainder(FIELD_MODULUS))
    assert torch.equal(combine_shares(seeded.shares), secret.remainder(FIELD_MODULUS))

    started = time.perf_counter()
    for _ in range(iterations):
        freivalds_challenge(secret, r=torch.ones(16, dtype=torch.int64))
    prime_ms = (time.perf_counter() - started) * 1000 / iterations
    return {
        "iterations": iterations,
        "matrix_shape": list(secret.shape),
        "field_modulus": FIELD_MODULUS,
        "prime_modulus": PRIME_MODULUS,
        "full_share_storage_bytes": full_storage,
        "seed_expanded_storage_bytes": seed_storage,
        "storage_savings_bytes": full_storage - seed_storage,
        "full_split_ms": full_ms,
        "seeded_split_ms": seeded_ms,
        "prime_challenge_ms": prime_ms,
        "full_share_security": ShareSecurity.INFORMATION_THEORETIC.value,
        "seed_expanded_security": ShareSecurity.COMPUTATIONAL.value,
        "exact_reconstruction": True,
        "measurement": "local-cpu-experiment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.iterations)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
