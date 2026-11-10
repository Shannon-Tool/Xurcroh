from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from xurcroh.algebra.attention import AttentionWeights, reference_attention
from xurcroh.algebra.feedforward import FeedForwardWeights, reference_feedforward
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
from xurcroh.shards.format import read_shard
from xurcroh.shards.keys import read_key_envelope


def test_compiled_attention_and_ffn_match_original(tiny_llama_path: Path, tmp_path: Path) -> None:
    master_key = bytes(range(32))
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            master_key=master_key,
            seed=223,
        )
    ).compile()
    original = load_file(str(tiny_llama_path / "model.safetensors"))
    key = read_key_envelope(result.key_envelopes[0], master_key)
    transformed, _ = read_shard(result.stage_shards[0], key)
    prefix = "model.layers.0"
    hidden = torch.randn(5, 16, dtype=torch.float64)
    basis = transformed["stage.input_basis"]
    gain = original[f"{prefix}.input_layernorm.weight"].double()
    original_attention = AttentionWeights(
        q=original[f"{prefix}.self_attn.q_proj.weight"].double() * gain,
        k=original[f"{prefix}.self_attn.k_proj.weight"].double() * gain,
        v=original[f"{prefix}.self_attn.v_proj.weight"].double() * gain,
        o=original[f"{prefix}.self_attn.o_proj.weight"].double(),
    )
    transformed_attention = AttentionWeights(
        q=transformed[f"{prefix}.self_attn.q_proj.weight"],
        k=transformed[f"{prefix}.self_attn.k_proj.weight"],
        v=transformed[f"{prefix}.self_attn.v_proj.weight"],
        o=transformed[f"{prefix}.self_attn.o_proj.weight"],
    )
    expected = reference_attention(hidden, original_attention, 4, 2, rotary=True, interleaved=False)
    actual = reference_attention(
        hidden @ basis.T, transformed_attention, 4, 2, rotary=True, interleaved=False
    )
    torch.testing.assert_close(actual @ basis, expected, atol=2e-10, rtol=2e-10)

    ffn_gain = original[f"{prefix}.post_attention_layernorm.weight"].double()
    original_ffn = FeedForwardWeights(
        gate=original[f"{prefix}.mlp.gate_proj.weight"].double() * ffn_gain,
        up=original[f"{prefix}.mlp.up_proj.weight"].double() * ffn_gain,
        down=original[f"{prefix}.mlp.down_proj.weight"].double(),
    )
    transformed_ffn = FeedForwardWeights(
        gate=transformed[f"{prefix}.mlp.gate_proj.weight"],
        up=transformed[f"{prefix}.mlp.up_proj.weight"],
        down=transformed[f"{prefix}.mlp.down_proj.weight"],
    )
    expected_ffn = reference_feedforward(hidden, original_ffn, "silu")
    actual_ffn = reference_feedforward(hidden @ basis.T, transformed_ffn, "silu")
    torch.testing.assert_close(actual_ffn @ basis, expected_ffn, atol=2e-10, rtol=2e-10)
