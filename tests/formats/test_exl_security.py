from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.formats.exl_utils import exl2_fixture

from xurcroh.errors import ModelLoadError
from xurcroh.formats import open_checkpoint


def test_exl2_rejects_invalid_bitrate(tmp_path: Path) -> None:
    _, _ = exl2_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.self_attn.q_proj.q_groups"] = torch.tensor(
        [7, 0, 4, 2], dtype=torch.int16
    )
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="bitrate"):
        reader.read_operation("model.layers.0.self_attn.q_proj")


def test_exl2_rejects_bad_permutation(tmp_path: Path) -> None:
    _, _ = exl2_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.self_attn.q_proj.q_invperm"] = torch.zeros(64, dtype=torch.int32)
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="permutation"):
        reader.read_operation("model.layers.0.self_attn.q_proj")
