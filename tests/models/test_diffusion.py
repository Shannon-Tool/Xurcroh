from __future__ import annotations

import torch

from xurcroh.algebra.attention import AttentionWeights, sample_orthogonal
from xurcroh.models.diffusion import (
    FlowEulerScheduler,
    JointAttentionWeights,
    LatentTransform,
    reference_joint_attention,
)


def test_latent_rotation_permutation_and_euler_step_are_equivariant() -> None:
    torch.manual_seed(277)
    basis = sample_orthogonal(8, seed=281)
    permutation = torch.randperm(5, generator=torch.Generator().manual_seed(283))
    transform = LatentTransform(basis, permutation)
    latent = torch.randn(5, 8, dtype=torch.float64)
    velocity = torch.randn(5, 8, dtype=torch.float64)
    transformed = transform.apply(latent)
    transformed_velocity = transform.apply(velocity)
    scheduler = FlowEulerScheduler([1.0, 0.5, 0.0])
    expected = latent - 0.5 * velocity
    actual = scheduler.step(transformed, transformed_velocity, 0)
    torch.testing.assert_close(transform.recover(actual), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(transform.recover(transformed), latent, atol=1e-12, rtol=1e-12)


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
