from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from xurcroh.fortress.arithmetic import freivalds_challenge, split_secret
from xurcroh.fortress.grid import RoleExecutionError, WeightShare, execute_grid


def test_grid_combines_every_role_and_verifies_each_product() -> None:
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

    assert torch.equal(result, (activation @ weight).remainder(2**32))


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
    shares[0] = replace(shares[0], matrix=shares[0].matrix + 1)

    with pytest.raises(RoleExecutionError, match="provider-0"):
        execute_grid(activation_shares, shares)
