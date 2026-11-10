from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from xurcroh.client.boundary import ClientBoundary
from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler


def test_client_boundary_embeds_and_evaluates_output_head(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    master_key = bytes(range(32))
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            master_key=master_key,
            seed=331,
        )
    ).compile()
    boundary = ClientBoundary(result.boundary_shard, master_key)
    token_ids = torch.tensor([2, 3, 4])
    original = {
        name: tensor.double()
        for name, tensor in load_file(str(tiny_llama_path / "model.safetensors")).items()
    }
    embedded = boundary.embed(token_ids)
    expected = original["model.embed_tokens.weight"][token_ids] @ boundary.input_basis.T
    torch.testing.assert_close(embedded, expected)

    hidden = torch.randn(3, 16, dtype=torch.float64)
    logits = boundary.logits(hidden @ boundary.output_basis.T)
    from xurcroh.algebra.rmsnorm import unit_rms_norm

    epsilon = float(boundary._tensors["final_norm.epsilon"])
    expected_logits = (
        unit_rms_norm(hidden, epsilon)
        @ (original["lm_head.weight"] * original["model.norm.weight"]).T
    )
    torch.testing.assert_close(logits, expected_logits, atol=2e-10, rtol=2e-10)
    boundary.close()
    assert boundary.closed
