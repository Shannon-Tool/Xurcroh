from __future__ import annotations

import pytest
import torch

from xurcroh.fortress.arithmetic import PRIME_MODULUS, split_secret
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.grid import RoleExecutionError
from xurcroh.fortress.transport import combine_role_results


def exact_products(operation, shares):  # type: ignore[no-untyped-def]
    return {
        (activation_index, weight_index): [
            int(value)
            for value in (
                torch.tensor(
                    [
                        sum(int(a) * int(b) for a, b in zip(row, activation.tolist(), strict=True))
                        % PRIME_MODULUS
                        for row in weight.matrix.tolist()
                    ]
                )
            ).tolist()
        ]
        for activation_index, activation in enumerate(shares)
        for weight_index, weight in operation.shares.items()
    }


def test_each_weight_share_carries_two_independent_challenges() -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64), threshold=2, slice_count=3
    )

    for share in operation.shares.values():
        assert len(share.challenges) == 2
        first, second = share.challenges
        assert first is share.challenge
        assert not torch.equal(first.r, second.r)


def test_result_crafted_against_one_challenge_is_rejected_by_the_second() -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64), threshold=2, slice_count=3
    )
    shares = split_secret(torch.tensor([5, 7], dtype=torch.int64), count=2).shares
    products = exact_products(operation, shares)
    providers = {key: f"role-{key}" for key in products}
    expected = (operation.quantized.values @ torch.tensor([5, 7])).remainder(PRIME_MODULUS)
    assert torch.equal(
        combine_role_results(shares, operation.shares, products, providers), expected
    )

    # An error orthogonal to the first challenge vector passes a single Freivalds check.
    target = (0, 0)
    r = operation.shares[0].challenge.r.tolist()
    forged = list(products[target])
    forged[0] = (forged[0] + r[1]) % PRIME_MODULUS
    forged[1] = (forged[1] - r[0]) % PRIME_MODULUS
    assert forged != products[target]

    with pytest.raises(RoleExecutionError, match="Freivalds check failed"):
        combine_role_results(shares, operation.shares, {**products, target: forged}, providers)
