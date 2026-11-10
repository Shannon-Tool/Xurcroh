from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from shards.identities import StageIdentities
from xurcroh.algebra.attention import AttentionWeights, reference_attention
from xurcroh.algebra.feedforward import FeedForwardWeights, reference_feedforward
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key


def test_compiled_attention_and_ffn_match_original(tiny_llama_path: Path, tmp_path: Path) -> None:
    identities = StageIdentities.create(1)
    result = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=223)
    original = load_file(str(tiny_llama_path / "model.safetensors"))
    manifest = json.loads(result.manifest_path.read_text())
    envelope = json.loads(result.key_envelopes[0].read_text())
    key = open_stage_key(
        envelope,
        identities.private_keys[0],
        manifest_digest=manifest["digest"],
        checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
    )
    transformed, _ = read_shard(result.stage_shards[0], key)
    prefix = "model.layers.0"
    hidden = torch.randn(5, 16, dtype=torch.float64)
    boundary, _ = read_shard(result.boundary_shard, identities.boundary_key)
    basis = boundary["basis.input"]
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
