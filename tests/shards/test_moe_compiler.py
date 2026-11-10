from __future__ import annotations

from pathlib import Path

import torch

from shards.identities import StageIdentities
from xurcroh.algebra.feedforward import FeedForwardWeights, reference_feedforward
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler


def make_tensors() -> dict[str, torch.Tensor]:
    d, d_ff, experts = 16, 12, 3
    generator = torch.Generator().manual_seed(227)
    tensors = {
        "model.layers.0.input_layernorm.weight": torch.randn(
            d, generator=generator, dtype=torch.float64
        ),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(
            d, generator=generator, dtype=torch.float64
        ),
        "model.layers.0.mlp.gate.weight": torch.randn(
            experts, d, generator=generator, dtype=torch.float64
        ),
    }
    for name in ("q_proj", "o_proj"):
        tensors[f"model.layers.0.self_attn.{name}.weight"] = torch.randn(
            d, d, generator=generator, dtype=torch.float64
        )
    for name in ("k_proj", "v_proj"):
        tensors[f"model.layers.0.self_attn.{name}.weight"] = torch.randn(
            8, d, generator=generator, dtype=torch.float64
        )
    for expert in range(experts):
        for name in ("gate_proj", "up_proj"):
            tensors[f"model.layers.0.mlp.experts.{expert}.{name}.weight"] = torch.randn(
                d_ff, d, generator=generator, dtype=torch.float64
            )
        tensors[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] = torch.randn(
            d, d_ff, generator=generator, dtype=torch.float64
        )
    return tensors


def test_moe_router_and_experts_preserve_permuted_mapping(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities = StageIdentities.create(1)
    compiler = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "out",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=229,
        )
    )
    tensors = make_tensors()
    basis = torch.eye(16, dtype=torch.float64)
    transformed = compiler._compile_layer(tensors, "model.layers.0", basis, 233)
    permutation = transformed["model.layers.0.mlp.expert_permutation"]
    x = torch.randn(5, 16, dtype=torch.float64)
    gain = tensors["model.layers.0.post_attention_layernorm.weight"]
    original_logits = (x * gain) @ tensors["model.layers.0.mlp.gate.weight"].T
    transformed_logits = x @ transformed["model.layers.0.mlp.gate.weight"].T
    torch.testing.assert_close(transformed_logits, original_logits[:, permutation])

    for new_index, original_index in enumerate(permutation.tolist()):
        original = FeedForwardWeights(
            gate=tensors[f"model.layers.0.mlp.experts.{original_index}.gate_proj.weight"] * gain,
            up=tensors[f"model.layers.0.mlp.experts.{original_index}.up_proj.weight"] * gain,
            down=tensors[f"model.layers.0.mlp.experts.{original_index}.down_proj.weight"],
        )
        actual = FeedForwardWeights(
            gate=transformed[f"model.layers.0.mlp.experts.{new_index}.gate_proj.weight"],
            up=transformed[f"model.layers.0.mlp.experts.{new_index}.up_proj.weight"],
            down=transformed[f"model.layers.0.mlp.experts.{new_index}.down_proj.weight"],
        )
        torch.testing.assert_close(
            reference_feedforward(x, actual, "silu"),
            reference_feedforward(x, original, "silu"),
            atol=3e-11,
            rtol=3e-11,
        )
