from __future__ import annotations

import torch

from xurcroh.algebra.attention import (
    AttentionWeights,
    reference_attention,
    reference_attention_cached,
)
from xurcroh.algebra.feedforward import FeedForwardWeights
from xurcroh.models.dense import DenseLayerWeights, DenseStageRunner


def test_cached_attention_and_stage_match_full_recomputation() -> None:
    generator = torch.Generator().manual_seed(421)
    weights = AttentionWeights(
        q=torch.randn(4, 4, generator=generator, dtype=torch.float64),
        k=torch.randn(2, 4, generator=generator, dtype=torch.float64),
        v=torch.randn(2, 4, generator=generator, dtype=torch.float64),
        o=torch.randn(4, 4, generator=generator, dtype=torch.float64),
    )
    hidden = torch.randn(3, 4, dtype=torch.float64, generator=generator)
    expected = reference_attention(hidden, weights, 2, 1, rotary=True, interleaved=False)
    prefill, cache = reference_attention_cached(hidden[:2], weights, 2, 1, interleaved=False)
    decoded, cache = reference_attention_cached(
        hidden[2:], weights, 2, 1, past=cache, interleaved=False
    )
    torch.testing.assert_close(torch.cat([prefill, decoded]), expected)
    assert cache[0].shape == (3, 1, 2)
    assert cache[1].shape == (3, 1, 2)

    layer = DenseLayerWeights(
        attention=weights,
        feedforward=FeedForwardWeights(
            gate=torch.randn(8, 4, generator=generator, dtype=torch.float64),
            up=torch.randn(8, 4, generator=generator, dtype=torch.float64),
            down=torch.randn(4, 8, generator=generator, dtype=torch.float64),
        ),
        input_gain=torch.ones(4, dtype=torch.float64),
        post_attention_gain=torch.ones(4, dtype=torch.float64),
    )
    runner = DenseStageRunner([([layer], torch.eye(4, dtype=torch.float64), None)])
    cached_prefill = runner.run_cached("request", hidden[:2])
    cached_decode = runner.run_cached("request", hidden[2:])
    torch.testing.assert_close(
        torch.cat([cached_prefill, cached_decode]),
        DenseStageRunner([([layer], torch.eye(4, dtype=torch.float64), None)]).run(hidden),
    )
    assert runner.cache_entries == 1
    runner.drop_request("request")
    assert runner.cache_entries == 0
