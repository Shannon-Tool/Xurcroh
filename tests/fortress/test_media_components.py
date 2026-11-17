from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from xurcroh.config import ModelDefinition
from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
from xurcroh.fortress.media_components import (
    compile_media_checkpoint,
    discover_media_operations,
)


def test_multi_denoiser_discovery_and_compilation_use_namespaced_operations(
    tmp_path: Path,
) -> None:
    for component in ("transformer", "transformer_2"):
        root = tmp_path / component
        root.mkdir()
        save_file(
            {"blocks.0.attn.to_q.weight": torch.randn(4, 4)},
            str(root / "diffusion_pytorch_model.safetensors"),
        )
    adapter = MediaArchitectureAdapter.from_checkpoint(tmp_path, "image")
    assert adapter.components == ("transformer", "transformer_2")
    discovered = discover_media_operations(tmp_path, adapter)
    assert [operation.identifier for operation in discovered] == [
        "transformer:blocks.0.attn.to_q.weight",
        "transformer_2:blocks.0.attn.to_q.weight",
    ]
    packaged = []
    compiled = compile_media_checkpoint(
        ModelDefinition(path=str(tmp_path), modality="image", engine="fortress"),
        adapter,
        operation_sink=lambda name, operation: packaged.append(name),
    )
    assert set(compiled) == {"transformer", "transformer_2"}
    assert set(compiled["transformer"]) == {"blocks.0.attn.to_q.weight"}
    assert packaged == [
        "transformer:blocks.0.attn.to_q.weight",
        "transformer_2:blocks.0.attn.to_q.weight",
    ]


def test_media_adapter_rejects_missing_denoiser(tmp_path: Path) -> None:
    import pytest

    from xurcroh.errors import ModelLoadError

    with pytest.raises(ModelLoadError, match="denoiser"):
        MediaArchitectureAdapter.from_checkpoint(tmp_path, "video")
