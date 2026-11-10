from __future__ import annotations

import torch

from xurcroh.models.adaptation import (
    BlockScalarAdaLN,
    LoRAConfig,
    LoRALinear,
    fit_block_scalar_adaln,
)


def test_lora_linear_matches_base_plus_low_rank_update() -> None:
    torch.manual_seed(313)
    base = torch.randn(6, 4)
    module = LoRALinear(base, LoRAConfig(rank=2, alpha=2.0, seed=317))
    with torch.no_grad():
        module.a.copy_(torch.randn_like(module.a))
        module.b.copy_(torch.randn_like(module.b))
    x = torch.randn(3, 4)
    expected = x @ base.T + x @ module.a.T @ module.b.T
    torch.testing.assert_close(module(x), expected)


def test_block_scalar_adaln_adaptation_reduces_real_training_loss() -> None:
    torch.manual_seed(319)
    hidden = torch.randn(64, 8)
    residual = torch.randn(64, 8)
    beta = torch.randn(8)
    target = hidden * 1.3 + beta - 0.4 * residual
    model = BlockScalarAdaLN(8)
    report = fit_block_scalar_adaln(
        model,
        hidden,
        residual,
        target,
        LoRAConfig(rank=1, alpha=1.0, learning_rate=0.03, steps=120, seed=323),
    )
    assert report.final_loss < report.initial_loss * 0.05
    assert report.steps == 120
