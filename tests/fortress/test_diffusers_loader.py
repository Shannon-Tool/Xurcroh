from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from xurcroh.fortress.diffusers_loader import compile_diffusers_component
from xurcroh.fortress.transport import encode_role_artifact


def test_real_diffusers_layout_loads_only_transformer_linear_layers(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(json.dumps({"model_type": "flux"}))
    weights = {
        "transformer_blocks.0.attn.to_q.weight": torch.randn(4, 4),
        "transformer_blocks.0.attn.to_k.weight": torch.randn(4, 4),
        "transformer_blocks.0.attn.to_v.weight": torch.randn(4, 4),
        "transformer_blocks.0.attn.to_out.0.weight": torch.randn(4, 4),
        "transformer_blocks.0.norm.norm.weight": torch.ones(4),
    }
    save_file(weights, str(transformer / "diffusion_pytorch_model.safetensors"))
    text_encoder = tmp_path / "text_encoder"
    text_encoder.mkdir()
    save_file({"model.weight": torch.randn(4, 4)}, str(text_encoder / "model.safetensors"))
    vae = tmp_path / "vae"
    vae.mkdir()
    save_file({"decoder.weight": torch.randn(4, 4)}, str(vae / "model.safetensors"))

    from xurcroh.fortress.diffusers_loader import discover_diffusers_operations

    discovered = discover_diffusers_operations(tmp_path, component="transformer")
    operations = compile_diffusers_component(tmp_path, component="transformer")
    assert [operation.identifier for operation in discovered] == list(operations)
    by_name = {operation.identifier: operation for operation in discovered}
    assert (
        by_name["transformer_blocks.0.attn.to_q.weight"].paired_with
        == "transformer_blocks.0.attn.to_k.weight"
    )
    assert (
        by_name["transformer_blocks.0.attn.to_k.weight"].paired_with
        == "transformer_blocks.0.attn.to_q.weight"
    )
    assert len(operations) == 4
    assert "transformer_blocks.0.norm.norm.weight" not in operations
    assert all(
        b"basis" not in encode_role_artifact(operation, 0, bytes(range(32)))
        for operation in operations.values()
    )
