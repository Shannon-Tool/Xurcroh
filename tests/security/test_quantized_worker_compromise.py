from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tests.architectures.utils import write_llama_exl2, write_llama_exl3, write_llama_gguf

from xurcroh.fortress.attacks import evaluate_share_reconstruction
from xurcroh.fortress.compiler import materialize_weight_shares
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.transport import (
    decode_role_artifact,
    encode_role_artifact,
)


@pytest.mark.parametrize("source_format", ["gguf", "exl2", "exl3"])
def test_assigned_quantized_role_has_only_its_weight_share_and_no_client_secrets(
    tiny_llama_path: Path, tmp_path: Path, source_format: str
) -> None:
    writers = {
        "gguf": write_llama_gguf,
        "exl2": write_llama_exl2,
        "exl3": write_llama_exl3,
    }
    checkpoint = writers[source_format](tiny_llama_path, tmp_path / source_format)
    compiled = compile_fortress_checkpoint(checkpoint, seed=731, load_model=False)
    name = next(iter(compiled.operations))
    operation = compiled.operations[name]
    encoded = encode_role_artifact(operation, 0, bytes(1216))
    document = json.loads(encoded)
    assert "input_basis" not in document and "output_basis" not in document
    assert "trellis" not in document and "q_weight" not in document
    artifact = decode_role_artifact(encoded, allow_unattested=True)
    assigned = artifact.matrix
    shares = materialize_weight_shares(operation)
    torch.testing.assert_close(assigned, shares[0], rtol=0, atol=0)
    report = evaluate_share_reconstruction(
        operation.quantized.require_values(),
        torch.arange(operation.quantized.require_values().shape[1]),
        shares,
        [
            torch.zeros(operation.quantized.require_values().shape[1], dtype=torch.int64),
            torch.zeros(operation.quantized.require_values().shape[1], dtype=torch.int64),
        ],
    )
    assert report.full_weight_collusion_exact is True
    assert report.single_weight_share_reconstructs is False


def test_quantized_role_cannot_derive_peer_or_client_boundary_tensors(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    checkpoint = write_llama_exl3(tiny_llama_path, tmp_path / "exl3")
    compiled = compile_fortress_checkpoint(checkpoint, seed=731, load_model=False)
    operation = next(iter(compiled.operations.values()))
    encoded = encode_role_artifact(operation, 1, bytes(1216))
    for secret in (
        compiled.embedding.numpy().tobytes(),
        compiled.final_norm.numpy().tobytes(),
        compiled.lm_head.numpy().tobytes(),
        operation.input_basis.permutation.numpy().tobytes(),
        operation.output_basis.permutation.numpy().tobytes(),
    ):
        assert secret not in encoded


def test_namespaced_media_artifact_contains_no_latent_or_vae_material(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from xurcroh.config import ModelDefinition
    from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
    from xurcroh.fortress.media_components import compile_media_checkpoint

    for component in ("transformer", "transformer_2"):
        root = tmp_path / component
        root.mkdir()
        save_file(
            {"blocks.0.attn.to_q.weight": torch.randn(4, 4)},
            str(root / "diffusion_pytorch_model.safetensors"),
        )
    definition = ModelDefinition(
        path=str(tmp_path), modality="image", engine="fortress", fortress_seed=731
    )
    adapter = MediaArchitectureAdapter.from_checkpoint(tmp_path, "image")
    compiled = compile_media_checkpoint(definition, adapter)
    for component, operations in compiled.items():
        for operation in operations.values():
            encoded = encode_role_artifact(operation, 0, bytes(1216))
            assert f"{component}:".encode() not in encoded
            assert b"latents" not in encoded
            assert b"vae" not in encoded
            assert b"permutation" not in encoded
