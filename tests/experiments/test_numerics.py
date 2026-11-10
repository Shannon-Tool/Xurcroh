from __future__ import annotations

import torch
from experiments.numerics import linear_equivalence


def test_reduced_precision_experiment_compares_equivalent_rotations() -> None:
    assert linear_equivalence(torch.float32)["max_absolute_error"] < 1e-4
    assert linear_equivalence(torch.bfloat16)["max_absolute_error"] < 0.1
    assert linear_equivalence(torch.float16)["max_absolute_error"] < 0.02


def test_reduced_precision_experiment_reports_error_against_original() -> None:
    bf16 = linear_equivalence(torch.bfloat16)
    fp8 = linear_equivalence(torch.float8_e4m3fn)
    assert 0 < bf16["max_reference_error"] < 0.2
    assert 0 < fp8["max_reference_error"] < 4
