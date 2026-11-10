from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    learning_rate: float = 0.01
    steps: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha <= 0 or self.learning_rate <= 0 or self.steps < 1:
            raise ValueError("LoRA rank, alpha, learning rate, and steps must be positive")


@dataclass(frozen=True)
class AdaptationReport:
    initial_loss: float
    final_loss: float
    steps: int


class LoRALinear(torch.nn.Module):
    def __init__(self, base_weight: torch.Tensor, config: LoRAConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("base_weight", base_weight.detach().clone())
        generator = torch.Generator(device=base_weight.device).manual_seed(config.seed)
        self.a = torch.nn.Parameter(
            torch.randn(config.rank, base_weight.shape[1], generator=generator) * 0.01
        )
        self.b = torch.nn.Parameter(torch.zeros(base_weight.shape[0], config.rank))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = hidden @ self.get_buffer("base_weight").T
        update = hidden @ self.a.T @ self.b.T
        return base + update * (self.config.alpha / self.config.rank)


class BlockScalarAdaLN(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(()))
        self.alpha = torch.nn.Parameter(torch.zeros(()))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return hidden * (1 + self.gamma) + self.beta + self.alpha * residual


def fit_block_scalar_adaln(
    model: BlockScalarAdaLN,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    config: LoRAConfig,
) -> AdaptationReport:
    model.train()
    with torch.no_grad():
        initial_loss = torch.mean((model(hidden, residual) - target).square()).item()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.steps):
        optimizer.zero_grad()
        loss = torch.mean((model(hidden, residual) - target).square())
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = torch.mean((model(hidden, residual) - target).square()).item()
    return AdaptationReport(initial_loss=initial_loss, final_loss=final_loss, steps=config.steps)
