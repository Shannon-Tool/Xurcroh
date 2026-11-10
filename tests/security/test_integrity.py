from __future__ import annotations

import torch

from xurcroh.integrity import FloatingPointChecker, FreivaldsVerifier


def test_freivalds_accepts_exact_product_and_detects_tampering() -> None:
    weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    x = torch.tensor([2, -1, 4], dtype=torch.int64)
    z = weight @ x
    verifier = FreivaldsVerifier()
    challenge = verifier.challenge(weight, r=torch.tensor([3, 7], dtype=torch.int64))
    assert verifier.verify(x, z, challenge)
    assert not verifier.verify(x, z + 1, challenge)


def test_floating_point_checker_uses_tolerances_not_probability_claims() -> None:
    checker = FloatingPointChecker(atol=1e-3, rtol=1e-3)
    reference = torch.tensor([1.0, 2.0])
    assert checker.verify(reference, torch.tensor([1.0001, 1.9999]))
    assert not checker.verify(reference, torch.tensor([1.1, 2.0]))
