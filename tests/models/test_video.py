from __future__ import annotations

import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.models.video import (
    cache_dit_difference_norm,
    chunked_video_decode,
    timestep_expert,
)


def test_timestep_expert_selection_is_content_independent() -> None:
    assert timestep_expert(0.9, threshold=0.5) == "high_noise"
    assert timestep_expert(0.1, threshold=0.5) == "low_noise"
    assert timestep_expert(0.9, threshold=0.5, prompt_hash=12345) == "high_noise"


def test_cache_dit_norm_is_orthogonally_invariant() -> None:
    basis = sample_orthogonal(8, seed=311)
    current = torch.randn(4, 8, dtype=torch.float64)
    previous = torch.randn(4, 8, dtype=torch.float64)
    original = cache_dit_difference_norm(current, previous)
    rotated = cache_dit_difference_norm(current @ basis.T, previous @ basis.T)
    torch.testing.assert_close(rotated, original, atol=1e-12, rtol=1e-12)


def test_chunked_video_decoder_uses_four_frame_chunks() -> None:
    latent = torch.arange(5 * 2, dtype=torch.float64).reshape(5, 2)
    chunks = []
    output = chunked_video_decode(latent, lambda chunk: (chunks.append(len(chunk)), chunk * 2)[1])
    assert chunks == [4, 1]
    torch.testing.assert_close(output, latent * 2)
