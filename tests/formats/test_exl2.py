from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.formats.exl_utils import exl2_fixture, gptq_fixture

from xurcroh.errors import ModelLoadError
from xurcroh.formats import open_checkpoint
from xurcroh.formats.reference_bridge import ReferenceDecoderBridge


def test_exl2_mixed_bitrates_and_activation_permutation(tmp_path: Path) -> None:
    _, expected = exl2_fixture(tmp_path)
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation("model.layers.0.self_attn.q_proj")
        assert reader.source_format == "exl2"
        assert "EXL2" in reader.encodings
        assert reader.fingerprint
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exl2_without_permutation_and_bounded_output_columns(tmp_path: Path) -> None:
    _, expected = exl2_fixture(tmp_path, permutation=False, output_columns=33, actual_columns=33)
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation(
            "model.layers.0.self_attn.q_proj.weight", columns=slice(0, 33)
        )
    assert actual.shape == (33, 64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gptq_with_group_index(tmp_path: Path) -> None:
    expected = gptq_fixture(tmp_path)
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation("model.layers.0.mlp.up_proj")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exl2_rejects_inconsistent_group_layout(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    _, _ = exl2_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.self_attn.q_proj.q_groups"] = torch.tensor(
        [3, 0, 2, 0], dtype=torch.int16
    )
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="group"):
        reader.read_operation("model.layers.0.self_attn.q_proj")


def test_reference_bridge_records_pinned_decoder_identity() -> None:
    identity = ReferenceDecoderBridge("exllamav2").identity()
    assert identity["package"] == "exllamav2"
    assert identity["version"] == "0.3.2"
    assert identity["revision"]
