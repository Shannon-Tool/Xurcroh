from __future__ import annotations

import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.algebra.mla import MLAWeights, conjugate_mla, reference_mla
from xurcroh.algebra.recurrent import delta_step, frame_delta_step, rotate_state


def test_mla_conjugation_preserves_scores_and_output() -> None:
    torch.manual_seed(59)
    d, rank, latent, rope = 8, 4, 3, 4
    weights = MLAWeights(
        dkv=torch.randn(latent, d, dtype=torch.float64),
        dq=torch.randn(rank, d, dtype=torch.float64),
        q_abs=torch.randn(latent, rank, dtype=torch.float64),
        o_abs=torch.randn(d, latent, dtype=torch.float64),
        qr=torch.randn(rope, rank, dtype=torch.float64),
        kr=torch.randn(rope, d, dtype=torch.float64),
    )
    basis = sample_orthogonal(d, seed=61)
    latent_basis = sample_orthogonal(latent, seed=67)
    query_transform = sample_orthogonal(rank, seed=71)
    rope_q, rope_k = torch.eye(rope, dtype=torch.float64), torch.eye(rope, dtype=torch.float64)
    transformed = conjugate_mla(
        weights,
        basis,
        basis,
        latent_basis,
        query_transform,
        rope_q,
        rope_k,
    )
    x = torch.randn(5, d, dtype=torch.float64)
    original_scores, original_output = reference_mla(x, weights)
    rotated_scores, rotated_output = reference_mla(x @ basis.T, transformed)
    torch.testing.assert_close(rotated_scores, original_scores, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(rotated_output @ basis, original_output, atol=2e-12, rtol=2e-12)


def test_delta_rule_state_rotation_matches_reference() -> None:
    torch.manual_seed(73)
    dv, dk = 3, 4
    state = torch.randn(dv, dk, dtype=torch.float64)
    key, query = torch.randn(dk, dtype=torch.float64), torch.randn(dk, dtype=torch.float64)
    value = torch.randn(dv, dtype=torch.float64)
    r, o = sample_orthogonal(dv, seed=79), sample_orthogonal(dk, seed=83)
    original_state, original_output = delta_step(state, key, value, query, 0.4, 0.9)
    rotated_state, rotated_output = delta_step(
        rotate_state(state, r, o),
        o @ key,
        r @ value,
        o @ query,
        0.4,
        0.9,
    )
    torch.testing.assert_close(
        rotated_state, rotate_state(original_state, r, o), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(rotated_output, r @ original_output, atol=1e-12, rtol=1e-12)


def test_frame_wise_video_delta_update_is_equivariant() -> None:
    torch.manual_seed(89)
    tokens, dv, dk = 5, 3, 4
    key = torch.randn(tokens, dk, dtype=torch.float64)
    value = torch.randn(tokens, dv, dtype=torch.float64)
    query = torch.randn(tokens, dk, dtype=torch.float64)
    beta = torch.rand(tokens, dtype=torch.float64) + 0.1
    state = torch.randn(dv, dk, dtype=torch.float64)
    r, o = sample_orthogonal(dv, seed=97), sample_orthogonal(dk, seed=101)
    original_state, original_output = frame_delta_step(state, key, value, query, beta)
    rotated_state, rotated_output = frame_delta_step(
        rotate_state(state, r, o),
        key @ o.T,
        value @ r.T,
        query @ o.T,
        beta,
    )
    torch.testing.assert_close(
        rotated_state, rotate_state(original_state, r, o), atol=2e-11, rtol=2e-11
    )
    torch.testing.assert_close(rotated_output, r @ original_output, atol=2e-11, rtol=2e-11)
