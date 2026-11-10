from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.algebra.fwht import fwht
from xurcroh.algebra.orthogonal import conjugate_linear


def linear_equivalence(dtype: torch.dtype) -> dict[str, float | str]:
    generator = torch.Generator().manual_seed(167)
    weight = torch.randn(64, 32, generator=generator)
    bias = torch.randn(64, generator=generator)
    hidden = torch.randn(16, 32, generator=generator)
    basis_in = sample_orthogonal(32, seed=173, dtype=torch.float32)
    basis_out = sample_orthogonal(64, seed=179, dtype=torch.float32)
    t_weight, t_bias = conjugate_linear(weight, bias, basis_in, basis_out)
    reference = hidden @ weight.T + bias
    original_reference = reference.clone()
    rotated = (hidden @ basis_in.T) @ t_weight.T + t_bias
    recovered = rotated
    if dtype != torch.float32:
        recovered = recovered.to(dtype).float()
        reference = (reference @ basis_out.T).to(dtype).float() @ basis_out
    recovered = recovered @ basis_out
    error = (recovered - reference).abs()
    reference_error = (recovered - original_reference).abs()
    return {
        "dtype": str(dtype),
        "max_absolute_error": error.max().item(),
        "mean_absolute_error": error.mean().item(),
        "max_reference_error": reference_error.max().item(),
        "mean_reference_error": reference_error.mean().item(),
    }


def fwht_benchmark(dimension: int, iterations: int) -> dict[str, float | int | str]:
    values = torch.randn(32, dimension, generator=torch.Generator().manual_seed(181))
    signs = torch.where(
        torch.rand(dimension, generator=torch.Generator().manual_seed(191)) < 0.5, -1.0, 1.0
    )
    for _ in range(5):
        fwht(values, signs)
    start = time.perf_counter()
    for _ in range(iterations):
        fwht(values, signs)
    elapsed = time.perf_counter() - start
    return {
        "dimension": dimension,
        "batch": 32,
        "iterations": iterations,
        "total_seconds": elapsed,
        "milliseconds_per_transform": elapsed * 1000 / iterations,
        "measurement": "local CPU; not a hardware target",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/numerical-results.json")
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    if hasattr(torch, "float8_e4m3fn"):
        dtypes.append(torch.float8_e4m3fn)
    report = {
        "evidence": "verified by local experiment",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "linear_conjugation": [linear_equivalence(dtype) for dtype in dtypes],
        "fwht": fwht_benchmark(4096, args.iterations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
