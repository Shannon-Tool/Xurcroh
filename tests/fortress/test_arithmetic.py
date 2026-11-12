from __future__ import annotations

import pytest
import torch

from xurcroh.fortress.arithmetic import (
    QuantizedTensor,
    ShareSecurity,
    combine_shares,
    freivalds_challenge,
    quantized_matmul,
    refresh_shares,
    seeded_split_secret,
    split_secret,
    symmetric_quantize,
    verify_freivalds,
)


def test_quantized_matmul_is_exact_and_rejects_ring_overflow() -> None:
    weight = torch.tensor([[-8, 7], [3, -2]], dtype=torch.int64)
    activation = torch.tensor([[11, -5], [2, 4]], dtype=torch.int64)
    quantized_weight = symmetric_quantize(weight, bits=4)
    quantized_activation = symmetric_quantize(activation, bits=12)

    product = quantized_matmul(quantized_activation, quantized_weight)

    expected = (quantized_activation.values - quantized_activation.zero_point) @ (
        quantized_weight.values - quantized_weight.zero_point
    )
    assert torch.equal(product.values, expected)
    assert product.scale == quantized_activation.scale * quantized_weight.scale
    with pytest.raises(OverflowError, match="ring"):
        quantized_matmul(
            QuantizedTensor(torch.full((1, 2), 2**30, dtype=torch.int64), 1.0, 0),
            QuantizedTensor(torch.full((2, 1), 7, dtype=torch.int64), 1.0, 0),
        )


def test_random_shares_reconstruct_and_refresh_without_changing_the_secret() -> None:
    secret = torch.tensor([[0, 1, 2**32 - 1, 17]], dtype=torch.int64)
    grid = split_secret(secret, count=3)
    shares = grid.shares

    assert len(shares) == 3
    assert grid.security is ShareSecurity.INFORMATION_THEORETIC
    assert torch.equal(combine_shares(shares), secret.remainder(2**32))
    refreshed = refresh_shares(shares, first=0, second=1)
    assert torch.equal(combine_shares(refreshed), secret.remainder(2**32))
    assert not torch.equal(refreshed[0], shares[0])


def test_seeded_shares_are_deterministic_and_computational_not_information_theoretic() -> None:
    secret = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    first = seeded_split_secret(secret, seed=bytes(range(32)), count=2)
    second = seeded_split_secret(secret, seed=bytes(range(32)), count=2)

    assert torch.equal(combine_shares(first.shares), secret.remainder(2**32))
    assert all(
        torch.equal(left, right) for left, right in zip(first.shares, second.shares, strict=False)
    )
    assert first.security is ShareSecurity.COMPUTATIONAL
    assert split_secret(secret, count=2).security is ShareSecurity.INFORMATION_THEORETIC


def test_freivalds_verification_accepts_exact_products_and_detects_tampering() -> None:
    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    activation = torch.tensor([5, 7], dtype=torch.int64)
    product = activation @ weight.T
    challenge = freivalds_challenge(weight, r=torch.tensor([11, 13], dtype=torch.int64))

    assert verify_freivalds(activation, product, challenge)
    assert not verify_freivalds(activation, product + 1, challenge)
