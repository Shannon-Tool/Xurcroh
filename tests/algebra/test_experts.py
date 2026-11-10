from __future__ import annotations

import torch

from xurcroh.algebra.moe import permute_experts


def test_expert_and_router_permutation_preserves_selection_mapping() -> None:
    router = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    experts = [torch.full((2, 3), float(index)) for index in range(4)]
    permutation = torch.tensor([2, 0, 3, 1])
    transformed_router, transformed_experts = permute_experts(router, experts, permutation)
    torch.testing.assert_close(transformed_router, router[permutation])
    for new_index, original_index in enumerate(permutation.tolist()):
        torch.testing.assert_close(transformed_experts[new_index], experts[original_index])


def test_expert_permutation_rejects_duplicate_indices() -> None:
    import pytest

    with pytest.raises(ValueError, match="every expert"):
        permute_experts(
            torch.ones(2, 3),
            [torch.ones(2, 3), torch.ones(2, 3)],
            torch.tensor([0, 0]),
        )
