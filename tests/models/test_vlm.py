from __future__ import annotations

import torch

from xurcroh.models.vlm import (
    DeepStackInjector,
    GemmaAudioEmbedder,
    GemmaVisionEmbedder,
    pad_with_chaff,
)


def test_gemma_merged_patchification_and_projection() -> None:
    torch.manual_seed(257)
    image = torch.randn(3, 96, 48)
    weight = torch.randn(8, 48 * 48 * 3, dtype=torch.float64)
    embedder = GemmaVisionEmbedder(weight)
    output = embedder.embed(image)
    expected = image.reshape(3, 2, 48, 1, 48).permute(1, 3, 0, 2, 4).reshape(2, -1)
    assert output.shape == (2, 8)
    torch.testing.assert_close(output, expected.to(weight.dtype) @ weight.T)


def test_gemma_audio_framing_pads_only_final_frame() -> None:
    waveform = torch.randn(1000, dtype=torch.float64)
    weight = torch.randn(8, 640, dtype=torch.float64)
    embedder = GemmaAudioEmbedder(weight)
    output = embedder.embed(waveform)
    from xurcroh.algebra.rmsnorm import unit_rms_norm

    padded = torch.cat([waveform, torch.zeros(280, dtype=torch.float64)])
    expected = unit_rms_norm(padded.reshape(2, 640), 1e-6) @ weight.T
    torch.testing.assert_close(output, expected)


def test_chaff_padding_uses_exact_attention_mask() -> None:
    embeddings = torch.randn(3, 8, dtype=torch.float64)
    mask = torch.ones(3, dtype=torch.bool)
    padded, padded_mask = pad_with_chaff(embeddings, mask, bucket_size=6, seed=263)
    assert padded.shape == (6, 8)
    assert padded_mask.tolist() == [True, True, True, False, False, False]
    torch.testing.assert_close(padded[:3], embeddings)
    assert not torch.equal(padded[3], padded[4])


def test_deepstack_connector_preserves_basis_transform() -> None:
    from xurcroh.algebra.attention import sample_orthogonal

    source = sample_orthogonal(8, seed=269)
    target = sample_orthogonal(8, seed=271)
    connector = torch.randn(8, 8, dtype=torch.float64)
    injector = DeepStackInjector([(2, connector, source, target)])
    visual = torch.randn(4, 8, dtype=torch.float64)
    hidden = torch.randn(4, 8, dtype=torch.float64)
    original = hidden @ target + visual @ connector.T
    actual = injector.inject(hidden, {2: visual @ source.T}, target_layer=2)
    torch.testing.assert_close(actual @ target, original, atol=1e-12, rtol=1e-12)
