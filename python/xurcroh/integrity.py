from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FreivaldsChallenge:
    r: torch.Tensor
    v: torch.Tensor


class FreivaldsVerifier:
    def __init__(self, *, prime: int = 2**32 - 5) -> None:
        self.prime = prime

    def challenge(self, weight: torch.Tensor, *, r: torch.Tensor) -> FreivaldsChallenge:
        if weight.ndim != 2 or r.shape != (weight.shape[0],):
            raise ValueError("challenge vector must match matrix output dimension")
        return FreivaldsChallenge(r=r, v=torch.remainder(r @ weight, self.prime))

    def verify(self, x: torch.Tensor, z: torch.Tensor, challenge: FreivaldsChallenge) -> bool:
        if x.shape[0] != challenge.v.shape[0] or z.shape[0] != challenge.r.shape[0]:
            raise ValueError("input/output dimensions do not match challenge")
        left = torch.remainder(x @ challenge.v, self.prime)
        right = torch.remainder(z @ challenge.r, self.prime)
        return bool(torch.equal(left, right))


@dataclass(frozen=True)
class FloatingPointChecker:
    atol: float
    rtol: float

    def verify(self, reference: torch.Tensor, actual: torch.Tensor) -> bool:
        return bool(torch.allclose(reference, actual, atol=self.atol, rtol=self.rtol))
