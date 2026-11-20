from __future__ import annotations

import torch

from xurcroh.algebra.attention import AttentionWeights
from xurcroh.models.diffusion import (
    FlowEulerScheduler,
    JointAttentionWeights,
    reference_joint_attention,
)


def test_flow_euler_scheduler_advances_by_schedule_delta() -> None:
    scheduler = FlowEulerScheduler([1.0, 0.5, 0.0])
    latent = torch.randn(5, 8, dtype=torch.float64)
    velocity = torch.randn(5, 8, dtype=torch.float64)
    torch.testing.assert_close(
        scheduler.step(latent, velocity, 0),
        latent - 0.5 * velocity,
        atol=1e-12,
        rtol=1e-12,
    )


def test_double_stream_joint_attention_shared_head_transform() -> None:
    torch.manual_seed(293)
    heads, head_dim = 2, 4
    image_weights = AttentionWeights(
        q=torch.randn(8, 8, dtype=torch.float64),
        k=torch.randn(8, 8, dtype=torch.float64),
        v=torch.randn(8, 8, dtype=torch.float64),
        o=torch.randn(8, 8, dtype=torch.float64),
    )
    weights = JointAttentionWeights(image_weights, image_weights)
    image = torch.randn(3, 8, dtype=torch.float64)
    text = torch.randn(2, 8, dtype=torch.float64)
    image_output, text_output = reference_joint_attention(image, text, weights, heads, head_dim)
    assert image_output.shape == (3, 8)
    assert text_output.shape == (2, 8)
    combined = torch.cat([image_output, text_output])
    single = reference_joint_attention(
        torch.cat([image, text]),
        torch.empty(0, 8, dtype=torch.float64),
        weights,
        heads,
        head_dim,
    )[0]
    torch.testing.assert_close(combined, single, atol=1e-12, rtol=1e-12)
