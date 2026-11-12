from __future__ import annotations

import torch

from xurcroh.algebra.attention import AttentionWeights
from xurcroh.algebra.feedforward import FeedForwardWeights
from xurcroh.fortress.combiner import FortressCombiner
from xurcroh.fortress.compiler import compile_operation
from xurcroh.models.dense import DenseLayerWeights


def matrix(generator: torch.Generator, rows: int, cols: int) -> torch.Tensor:
    return torch.randint(-2, 3, (rows, cols), generator=generator, dtype=torch.int64)


def build_combiner() -> FortressCombiner:
    generator = torch.Generator().manual_seed(841)
    layers = []
    operations = {}
    for index in range(2):
        attention = AttentionWeights(
            q=matrix(generator, 4, 4),
            k=matrix(generator, 4, 4),
            v=matrix(generator, 4, 4),
            o=matrix(generator, 4, 4),
            q_norm=torch.ones(2, dtype=torch.int64),
            k_norm=torch.ones(2, dtype=torch.int64),
        )
        feedforward = FeedForwardWeights(
            gate=matrix(generator, 8, 4),
            up=matrix(generator, 8, 4),
            down=matrix(generator, 4, 8),
        )
        layers.append(
            DenseLayerWeights(
                attention=attention,
                feedforward=feedforward,
                input_gain=torch.ones(4, dtype=torch.int64),
                post_attention_gain=torch.ones(4, dtype=torch.int64),
            )
        )
        prefix = f"model.layers.{index}"
        for name, weight in (
            ("self_attn.q_proj", attention.q),
            ("self_attn.k_proj", attention.k),
            ("self_attn.v_proj", attention.v),
            ("self_attn.o_proj", attention.o),
            ("mlp.gate_proj", feedforward.gate),
            ("mlp.up_proj", feedforward.up),
            ("mlp.down_proj", feedforward.down),
        ):
            operations[f"{prefix}.{name}"] = compile_operation(weight, threshold=2, slice_count=3)
    return FortressCombiner(
        layers=layers,
        operations=operations,
        query_heads=2,
        kv_heads=2,
        final_norm=torch.ones(4, dtype=torch.int64),
        lm_head=matrix(generator, 16, 4),
    )


def test_combiner_matches_quantized_reference_for_prefill_decode_and_sampling() -> None:
    combiner = build_combiner()
    hidden = torch.tensor([[1, 0, -1, 2], [0, 2, 1, -1], [2, 1, 0, 1]], dtype=torch.float64)
    expected = combiner.reference_forward(hidden)

    actual = combiner.run(hidden)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    prefill = combiner.run_cached("session:request", hidden[:2])
    decoded = combiner.run_cached("session:request", hidden[2:])
    torch.testing.assert_close(torch.cat([prefill, decoded]), expected, atol=1e-7, rtol=1e-7)

    token = combiner.sample(actual[-1:])
    assert token.item() == combiner.reference_sample(expected[-1:]).item()
    combiner.drop_request("session:request")
    assert combiner.cache_entries == 0
