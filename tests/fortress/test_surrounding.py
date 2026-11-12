from __future__ import annotations

import pytest
import torch

from xurcroh.fortress.arithmetic import split_secret
from xurcroh.fortress.attacks import evaluate_share_reconstruction
from xurcroh.fortress.canary import CanaryRegistry, DecoyScheduler


def test_decoys_are_mixed_with_real_calls_at_a_fixed_total_load() -> None:
    scheduler = DecoyScheduler()
    plan = scheduler.plan(real_calls=7, total_calls=20)
    assert len(plan) == 20
    assert sum(plan) == 7
    assert sum(not value for value in plan) == 13
    with pytest.raises(ValueError):
        scheduler.plan(real_calls=21, total_calls=20)


def test_canary_trip_revokes_only_attributed_roles_and_is_idempotent() -> None:
    revoked: list[str] = []
    registry = CanaryRegistry(revoke=revoked.append)
    secret = bytes(range(64))
    canary_id = registry.register(secret, observers=("role-0-1",))
    assert secret not in repr(registry).encode()
    assert registry.observe(canary_id, bytes(64)) is False
    assert revoked == []
    assert registry.observe(canary_id, secret) is True
    assert registry.observe(canary_id, secret) is False
    assert revoked == ["role-0-1"]
    assert registry.trips[canary_id].observers == ("role-0-1",)


def test_attack_report_demonstrates_collusion_boundary_without_overclaiming() -> None:
    weight = torch.tensor([[1, 2], [3, 4]])
    activation = torch.tensor([5, 7])
    weight_shares = split_secret(weight, count=2).shares
    activation_shares = split_secret(activation, count=2).shares
    report = evaluate_share_reconstruction(weight, activation, weight_shares, activation_shares)
    assert report.full_weight_collusion_exact
    assert report.full_activation_collusion_exact
    assert not report.single_weight_share_reconstructs
    assert not report.single_activation_share_reconstructs
    assert report.missing_weight_share_search_bits == 128
    assert report.missing_activation_share_search_bits == 64
