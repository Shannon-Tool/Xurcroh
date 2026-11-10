from __future__ import annotations

import torch

from xurcroh.models.moe import ConstantLoadDispatcher, reference_moe


def test_constant_load_dispatch_preserves_outputs_and_packet_sizes() -> None:
    torch.manual_seed(251)
    tokens, hidden_size, experts = 7, 4, 5
    hidden = torch.randn(tokens, hidden_size, dtype=torch.float64)
    router = torch.randn(experts, hidden_size, dtype=torch.float64)
    dispatcher = ConstantLoadDispatcher(top_k=2, constant_load=6)
    plan = dispatcher.dispatch(hidden, router)
    assert plan.loads.tolist() == [6] * experts
    assert plan.dummy_mask.sum().item() > 0
    assert torch.all(plan.weights[plan.dummy_mask] == 0)

    expert_outputs = torch.randn(experts, 6, hidden_size, dtype=torch.float64)
    actual = dispatcher.combine(plan, expert_outputs)
    expected = reference_moe(hidden, router, expert_outputs, plan, top_k=2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_adversarial_single_expert_route_is_dropless_and_constant_load() -> None:
    hidden = torch.randn(9, 4, dtype=torch.float64)
    router = torch.zeros(3, 4, dtype=torch.float64)
    dispatcher = ConstantLoadDispatcher(top_k=1, constant_load=9)
    plan = dispatcher.dispatch(hidden, router)
    assert plan.loads.tolist() == [9, 9, 9]
    assert (~plan.dummy_mask).sum().item() == 9
    assert torch.all(plan.token_indices[plan.dummy_mask] == 0)


def test_insufficient_capacity_fails_instead_of_dropping_tokens() -> None:
    import pytest

    dispatcher = ConstantLoadDispatcher(top_k=2, constant_load=2)
    with pytest.raises(ValueError, match="capacity"):
        dispatcher.dispatch(torch.randn(5, 4), torch.randn(3, 4))
