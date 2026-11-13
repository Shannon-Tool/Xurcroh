from __future__ import annotations

import pytest
import torch

from xurcroh.fortress.arithmetic import FIELD_MODULUS, freivalds_challenge, split_secret
from xurcroh.fortress.grid import RoleExecutionError, WeightShare, execute_grid
from xurcroh.fortress.transport import combine_role_results


def test_grid_combines_every_role() -> None:
    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    activation = torch.tensor([5, 7], dtype=torch.int64)
    weight_shares = split_secret(weight, count=2).shares
    activation_shares = split_secret(activation, count=2).shares
    shares = {
        index: WeightShare(
            weight_index=index,
            matrix=matrix,
            challenge=freivalds_challenge(matrix, r=torch.tensor([11, 13], dtype=torch.int64)),
            provider_id=f"provider-{index}",
        )
        for index, matrix in enumerate(weight_shares)
    }

    result = execute_grid(activation_shares, shares, replica_fraction=1.0)

    assert torch.equal(result, (activation @ weight.T).remainder(FIELD_MODULUS))


def test_grid_attributes_invalid_products_to_the_responsible_role() -> None:
    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    activation = torch.tensor([5, 7], dtype=torch.int64)
    weight_shares = split_secret(weight, count=2).shares
    activation_shares = split_secret(activation, count=2).shares
    shares = {
        index: WeightShare(
            weight_index=index,
            matrix=matrix,
            challenge=freivalds_challenge(matrix, r=torch.tensor([11, 13], dtype=torch.int64)),
            provider_id=f"provider-{index}",
        )
        for index, matrix in enumerate(weight_shares)
    }
    results = {
        (activation_index, weight_index): [
            sum(
                int(left) * int(right)
                for left, right in zip(activation_shares[activation_index], row, strict=True)
            )
            % FIELD_MODULUS
            for row in share.matrix
        ]
        for activation_index in range(2)
        for weight_index, share in shares.items()
    }
    providers = {key: f"provider-{key[1]}" for key in results}
    results[(0, 0)][0] += 1

    with pytest.raises(RoleExecutionError, match="provider-0"):
        combine_role_results(activation_shares, shares, results, providers)
