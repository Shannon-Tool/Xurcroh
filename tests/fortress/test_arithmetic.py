from __future__ import annotations

import pytest
import torch

from xurcroh.fortress.arithmetic import (
    MAX_SHARE_COUNT,
    MAX_SIGNED_FIELD_VALUE,
    PRIME_MODULUS,
    QuantizedTensor,
    ShareSecurity,
    check_unreduced_product_bound,
    combine_shares,
    freivalds_challenge,
    quantized_matmul,
    refresh_shares,
    seeded_split_secret,
    split_secret,
    symmetric_quantize,
    verify_freivalds,
)


def test_unreduced_product_bound_rejects_configurations_past_signed_field_limit() -> None:
    activation_bound = 2 ** (12 - 1)
    weight_bound = 2 ** (4 - 1)
    largest_inner = MAX_SIGNED_FIELD_VALUE // (activation_bound * weight_bound)
    check_unreduced_product_bound(largest_inner, activation_bound, weight_bound)
    with pytest.raises(OverflowError, match="field"):
        check_unreduced_product_bound(largest_inner + 1, activation_bound, weight_bound)


def test_combiner_rejects_ring_overflow_before_grid_dispatch() -> None:
    from xurcroh.fortress.combiner import FortressCombiner
    from xurcroh.fortress.compiler import compile_operation

    weight = torch.full((1, 150_000), 2 ** (4 - 1) - 1, dtype=torch.int64)
    operation = compile_operation(weight, threshold=2, slice_count=3)
    combiner = FortressCombiner(
        layers=[],
        operations={"op": operation},
        query_heads=1,
        kv_heads=1,
        final_norm=torch.ones(1, dtype=torch.float64),
        lm_head=torch.ones(1, 1, dtype=torch.float64),
    )
    with pytest.raises(OverflowError, match="field"):
        combiner._linear("op", torch.zeros(150_000, dtype=torch.float64))


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
    with pytest.raises(OverflowError, match="field"):
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
    assert torch.equal(combine_shares(shares), secret.remainder(PRIME_MODULUS))
    refreshed = refresh_shares(shares, first=0, second=1)
    assert torch.equal(combine_shares(refreshed), secret.remainder(PRIME_MODULUS))
    assert not torch.equal(refreshed[0], shares[0])


def test_share_count_above_accumulation_bound_is_rejected() -> None:
    secret = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="accumulation bound"):
        split_secret(secret, count=MAX_SHARE_COUNT + 1)
    with pytest.raises(ValueError, match="accumulation bound"):
        seeded_split_secret(secret, seed=bytes(range(32)), count=MAX_SHARE_COUNT + 1)
    with pytest.raises(ValueError, match="accumulation bound"):
        combine_shares([secret] * (MAX_SHARE_COUNT + 1))


def test_seeded_shares_are_deterministic_and_computational_not_information_theoretic() -> None:
    secret = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    first = seeded_split_secret(secret, seed=bytes(range(32)), count=3)
    second = seeded_split_secret(secret, seed=bytes(range(32)), count=3)

    assert torch.equal(combine_shares(first.shares), secret.remainder(PRIME_MODULUS))
    assert not torch.equal(first.shares[0], first.shares[1])
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


def test_random_and_seeded_shares_use_canonical_field_elements() -> None:
    secret = torch.tensor([[0, 1, PRIME_MODULUS - 1, 17]], dtype=torch.int64)
    for grid in (
        split_secret(secret, count=3, modulus=PRIME_MODULUS),
        seeded_split_secret(secret, seed=bytes(range(32)), count=3),
    ):
        for share in grid.shares:
            assert bool(torch.all((share >= 0) & (share < PRIME_MODULUS)))
        assert torch.equal(combine_shares(grid.shares, modulus=PRIME_MODULUS), secret)


def test_low_32_bit_truncation_is_not_a_valid_field_product() -> None:
    activation = torch.tensor([PRIME_MODULUS - 1], dtype=torch.int64)
    weight = torch.tensor([[2]], dtype=torch.int64)
    product = (activation @ weight.T).remainder(PRIME_MODULUS)
    truncated = (activation @ weight.T).remainder(2**32)
    challenge = freivalds_challenge(weight, r=torch.tensor([7], dtype=torch.int64))
    assert verify_freivalds(activation, product, challenge)
    assert not torch.equal(product, truncated)
    assert not verify_freivalds(activation, truncated, challenge)


def test_secure_uniform_reduces_draws_above_the_signed_range_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 64-bit draw at or above 2**63 must be reduced as the unsigned value it is.

    Reducing it after a signed conversion maps it to a different residue, which makes
    some share values very slightly more likely than others.
    """

    import numpy as np

    from xurcroh.fortress import arithmetic

    limit = (2**64 // arithmetic.FIELD_MODULUS) * arithmetic.FIELD_MODULUS
    draws = [0, 7, 2**63 - 1, 2**63, 2**63 + 7, limit - 1]
    monkeypatch.setattr(
        arithmetic.os,
        "urandom",
        lambda size: np.array(draws, dtype=np.uint64).tobytes()[:size],
    )

    values = arithmetic._secure_uniform((len(draws),), arithmetic.FIELD_MODULUS)

    assert values.tolist() == [draw % arithmetic.FIELD_MODULUS for draw in draws]


def test_secure_uniform_rejects_draws_that_would_bias_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from xurcroh.fortress import arithmetic

    limit = (2**64 // arithmetic.FIELD_MODULUS) * arithmetic.FIELD_MODULUS
    batches = iter([[limit, 2**64 - 1, 5], [limit + 1, 9], [11]])
    monkeypatch.setattr(
        arithmetic.os,
        "urandom",
        lambda size: np.array(next(batches), dtype=np.uint64).tobytes()[:size],
    )

    assert arithmetic._secure_uniform((3,), arithmetic.FIELD_MODULUS).tolist() == [5, 9, 11]


def test_secure_uniform_supports_a_power_of_two_modulus() -> None:
    from xurcroh.fortress import arithmetic

    values = arithmetic._secure_uniform((64,), arithmetic.RING_MODULUS)

    assert values.shape == (64,)
    assert bool(((values >= 0) & (values < arithmetic.RING_MODULUS)).all())
