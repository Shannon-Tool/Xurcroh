from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tests.formats.exl3_utils import exl3_fixture
from tests.formats.exl_utils import exl2_fixture

from xurcroh.formats.exl2 import decode_exl2
from xurcroh.formats.exl3 import decode_exl3
from xurcroh.formats.reference_bridge import official_exl2_error, official_exl3_error

pytestmark = pytest.mark.hardware


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("official ExLlama differential validation requires CUDA")


def test_exl2_matches_official_cuda_reconstruction(tmp_path: Path) -> None:
    _require_cuda()
    _, expected = exl2_fixture(tmp_path)
    tensors = load_file(tmp_path / "model.safetensors")
    prefix = "model.layers.0.self_attn.q_proj"
    assert official_exl2_error(tensors, prefix, expected) == 0
    torch.testing.assert_close(decode_exl2(tensors, prefix), expected, rtol=0, atol=0)


def test_exl3_matches_official_cuda_reconstruction(tmp_path: Path) -> None:
    _require_cuda()
    expected = exl3_fixture(tmp_path, 2, "3inst")
    tensors = load_file(tmp_path / "model.safetensors")
    prefix = "model.layers.0.mlp.down_proj"
    assert official_exl3_error(tensors, prefix, expected) == 0
    torch.testing.assert_close(decode_exl3(tensors, prefix), expected, rtol=0, atol=0)
