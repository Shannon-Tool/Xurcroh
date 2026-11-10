from __future__ import annotations

from collections.abc import Sequence

import torch


def permute_experts(
    router: torch.Tensor,
    experts: Sequence[torch.Tensor],
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    count = len(experts)
    if router.ndim != 2 or router.shape[0] != count:
        raise ValueError("router rows must match the number of experts")
    if permutation.shape != (count,) or not torch.equal(
        torch.sort(permutation.cpu()).values, torch.arange(count)
    ):
        raise ValueError("expert permutation must contain every expert exactly once")
    permutation = permutation.to(router.device)
    transformed_experts = [experts[index] for index in permutation.cpu().tolist()]
    return router[permutation], transformed_experts
