from __future__ import annotations

import torch

from xurcroh.algebra.attention import rope_commuting_pair, sample_orthogonal
from xurcroh.algebra.orthogonal import basis_transition
from xurcroh.algebra.positional import apply_interleaved_mrope, interleaved_frequencies
from xurcroh.algebra.rmsnorm import homogenize_embedding_norms


def test_interleaved_mrope_commutes_with_contragredient_qk() -> None:
    torch.manual_seed(139)
    head_dimension = 8
    q_transform, k_transform = rope_commuting_pair(1, 1, head_dimension, seed=149)
    positions = torch.tensor([[2, 3, 5], [7, 11, 13]])
    frequencies = interleaved_frequencies(head_dimension, base=10_000)
    q = torch.randn(2, 1, head_dimension, dtype=torch.float64)
    k = torch.randn(2, 1, head_dimension, dtype=torch.float64)
    q_hat = q @ q_transform.T
    k_hat = k @ k_transform.T
    original = (
        apply_interleaved_mrope(q, positions, frequencies)[:, 0]
        @ apply_interleaved_mrope(k, positions, frequencies)[:, 0].T
    )
    rotated = (
        apply_interleaved_mrope(q_hat, positions, frequencies)[:, 0]
        @ apply_interleaved_mrope(k_hat, positions, frequencies)[:, 0].T
    )
    torch.testing.assert_close(rotated, original, atol=2e-12, rtol=2e-12)


def test_mrope_secret_position_shift_cancels_in_relative_scores() -> None:
    torch.manual_seed(151)
    positions = torch.tensor([[2, 3, 5], [7, 11, 13]])
    shift = torch.tensor([19, 23, 29])
    q = torch.randn(2, 1, 8, dtype=torch.float64)
    frequencies = interleaved_frequencies(8, base=10_000)
    original = apply_interleaved_mrope(q, positions, frequencies)
    shifted = apply_interleaved_mrope(q, positions + shift, frequencies)
    torch.testing.assert_close(
        shifted[0] @ shifted[1].T,
        original[0] @ original[1].T,
        atol=2e-12,
        rtol=2e-12,
    )


def test_stage_basis_transition_and_embedding_homogenization() -> None:
    previous = sample_orthogonal(8, seed=157)
    following = sample_orthogonal(8, seed=163)
    transition = basis_transition(previous, following)
    x = torch.randn(4, 8, dtype=torch.float64)
    torch.testing.assert_close(
        (x @ previous.T) @ transition.T,
        x @ following.T,
        atol=1e-12,
        rtol=1e-12,
    )
    embeddings = torch.randn(20, 8, dtype=torch.float64) * torch.rand(20, 1, dtype=torch.float64)
    normalized = homogenize_embedding_norms(embeddings)
    torch.testing.assert_close(
        normalized.norm(dim=-1),
        torch.full((20,), 8**0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
