from __future__ import annotations

from dataclasses import dataclass

import torch

from xurcroh.algebra.attention import AttentionWeights
from xurcroh.algebra.feedforward import FeedForwardWeights, reference_feedforward
from xurcroh.algebra.modulation import adaln
from xurcroh.algebra.rmsnorm import unit_rms_norm
from xurcroh.models.diffusion import (
    FlowEulerScheduler,
    JointAttentionWeights,
    reference_joint_attention,
)


@dataclass(frozen=True)
class ModulationWeights:
    gamma: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor


@dataclass(frozen=True)
class DiTBlockWeights:
    image: AttentionWeights
    text: AttentionWeights
    image_ffn: FeedForwardWeights
    text_ffn: FeedForwardWeights
    image_modulation: ModulationWeights
    text_modulation: ModulationWeights
    epsilon: float = 1e-6


@dataclass(frozen=True)
class DiTStageWeights:
    blocks: tuple[DiTBlockWeights, ...]
    input: torch.Tensor
    output: torch.Tensor
    timesteps: tuple[float, ...]
    query_heads: int
    head_dimension: int
    latent_positions: int


def reference_dit_block(
    image: torch.Tensor,
    text: torch.Tensor,
    weights: DiTBlockWeights,
    query_heads: int,
    head_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_attention, text_attention = reference_joint_attention(
        image,
        text,
        JointAttentionWeights(weights.image, weights.text),
        query_heads,
        head_dimension,
    )
    image_residual = adaln(
        image,
        weights.image_modulation.gamma,
        weights.image_modulation.beta,
        weights.image_modulation.alpha,
        image_attention,
    )
    text_residual = adaln(
        text,
        weights.text_modulation.gamma,
        weights.text_modulation.beta,
        weights.text_modulation.alpha,
        text_attention,
    )
    image_output = image_residual + reference_feedforward(
        unit_rms_norm(image_residual, weights.epsilon), weights.image_ffn, "silu"
    )
    text_output = text_residual + reference_feedforward(
        unit_rms_norm(text_residual, weights.epsilon), weights.text_ffn, "silu"
    )
    return image_output, text_output


def dit_velocity(
    weights: DiTStageWeights, latent: torch.Tensor, text: torch.Tensor
) -> torch.Tensor:
    image = latent @ weights.input.T
    conditioning = text
    for block in weights.blocks:
        image, conditioning = reference_dit_block(
            image,
            conditioning,
            block,
            weights.query_heads,
            weights.head_dimension,
        )
    return image @ weights.output.T


def reference_dit_stage(
    weights: DiTStageWeights,
    latent: torch.Tensor,
    text: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    scheduler = FlowEulerScheduler(list(weights.timesteps))
    current = latent
    for index in range(steps):
        velocity = dit_velocity(weights, current, text)
        current = scheduler.step(current, velocity, index)
    return current
