from __future__ import annotations

import torch

from xurcroh.algebra.attention import (
    AttentionWeights,
    conjugate_attention,
    reference_attention,
    rope_commuting_pair,
    sample_orthogonal,
)


def test_gqa_rope_contragredient_attention_matches_reference() -> None:
    torch.manual_seed(19)
    d, h_q, h_kv, d_h, tokens = 16, 4, 2, 4, 5
    weights = AttentionWeights(
        q=torch.randn(h_q * d_h, d, dtype=torch.float64),
        k=torch.randn(h_kv * d_h, d, dtype=torch.float64),
        v=torch.randn(h_kv * d_h, d, dtype=torch.float64),
        o=torch.randn(d, h_q * d_h, dtype=torch.float64),
    )
    q_transform, k_transform = rope_commuting_pair(h_q, h_kv, d_h, seed=23)
    kv_blocks = [sample_orthogonal(d_h, seed=29 + i) for i in range(h_kv)]
    value_transform = torch.block_diag(*kv_blocks)
    output_value_transform = torch.block_diag(*[kv_blocks[i // (h_q // h_kv)] for i in range(h_q)])
    basis = sample_orthogonal(d, seed=31)
    transformed = conjugate_attention(
        weights,
        basis,
        basis,
        q_transform,
        k_transform,
        value_transform,
        output_value_transform.T,
    )
    x = torch.randn(tokens, d, dtype=torch.float64)
    original = reference_attention(x, weights, h_q, h_kv, rotary=True)
    rotated = reference_attention(x @ basis.T, transformed, h_q, h_kv, rotary=True)
    torch.testing.assert_close(rotated @ basis, original, atol=2e-11, rtol=2e-11)


def test_non_commuting_dense_qk_rotation_is_rejected() -> None:
    from xurcroh.algebra.attention import validate_rope_commuting

    with __import__("pytest").raises(ValueError, match="2x2"):
        validate_rope_commuting(sample_orthogonal(8, seed=37))


def test_qwen_qk_norm_rotation_preserves_attention() -> None:
    torch.manual_seed(41)
    d, h_q, h_kv, d_h, tokens = 16, 4, 2, 4, 5
    weights = AttentionWeights(
        q=torch.randn(h_q * d_h, d, dtype=torch.float64),
        k=torch.randn(h_kv * d_h, d, dtype=torch.float64),
        v=torch.randn(h_kv * d_h, d, dtype=torch.float64),
        o=torch.randn(d, h_q * d_h, dtype=torch.float64),
        q_norm=torch.rand(d_h, dtype=torch.float64) + 0.5,
        k_norm=torch.rand(d_h, dtype=torch.float64) + 0.5,
    )
    q_transform, k_transform = rope_commuting_pair(
        h_q,
        h_kv,
        d_h,
        seed=43,
        minimum_scale=1,
        maximum_scale=1,
        quarter_turns=True,
        shared_heads=True,
    )
    kv_blocks = [sample_orthogonal(d_h, seed=47 + i) for i in range(h_kv)]
    value_transform = torch.block_diag(*kv_blocks)
    output_value_transform = torch.block_diag(*[kv_blocks[i // 2] for i in range(h_q)])
    basis = sample_orthogonal(d, seed=53)
    transformed = conjugate_attention(
        weights,
        basis,
        basis,
        q_transform,
        k_transform,
        value_transform,
        output_value_transform.T,
    )
    x = torch.randn(tokens, d, dtype=torch.float64)
    original = reference_attention(x, weights, h_q, h_kv, rotary=True)
    rotated = reference_attention(x @ basis.T, transformed, h_q, h_kv, rotary=True)
    torch.testing.assert_close(rotated @ basis, original, atol=2e-11, rtol=2e-11)
