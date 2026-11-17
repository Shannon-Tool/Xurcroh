from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.formats.exl3_utils import exl3_fixture

from xurcroh.errors import ModelLoadError
from xurcroh.formats import open_checkpoint
from xurcroh.formats.reference_bridge import ReferenceDecoderBridge


@pytest.mark.parametrize(
    ("bitrate", "codebook"),
    [
        (1, "3inst"),
        (2, "3inst"),
        (4, "mcg"),
        (3, "mul1"),
        (1.5, "mul1"),
        (2.5, "mul1"),
        (3.5, "mul1"),
        (5, "3inst"),
        (6, "mcg"),
        (7, "mul1"),
        (8, "3inst"),
    ],
)
def test_exl3_trellis_codebooks_and_hadamard_restoration(
    tmp_path: Path, bitrate: float, codebook: str
) -> None:
    expected = exl3_fixture(tmp_path, bitrate, codebook)
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation("model.layers.0.mlp.down_proj")
        assert reader.source_format == "exl3"
        assert "EXL3" in reader.encodings
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exl3_bounded_output_column_slice(tmp_path: Path) -> None:
    expected = exl3_fixture(tmp_path, 2, "3inst")
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation("model.layers.0.mlp.down_proj", columns=slice(16, 48))
    torch.testing.assert_close(actual, expected[16:48], rtol=0, atol=0)


def test_exl3_rejects_fractional_rate_without_mul1(tmp_path: Path) -> None:
    _ = exl3_fixture(tmp_path, 1.5, "mul1")
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    del tensors["model.layers.0.mlp.down_proj.mul1"]
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="fractional"):
        reader.read_operation("model.layers.0.mlp.down_proj")


def test_exl3_rejects_conflicting_codebook_markers(tmp_path: Path) -> None:
    _ = exl3_fixture(tmp_path, 2, "mcg")
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.mlp.down_proj.mul1"] = torch.tensor(1, dtype=torch.int32)
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="codebook"):
        reader.read_operation("model.layers.0.mlp.down_proj")


def test_exl3_reference_identity() -> None:
    identity = ReferenceDecoderBridge("exllamav3").identity()
    assert identity["package"] == "exllamav3"
    assert identity["version"] == "1.6.7"
    assert identity["revision"]


def test_exl3_packed_sign_vectors(tmp_path: Path) -> None:
    expected = exl3_fixture(tmp_path, 2, "3inst")
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    prefix = "model.layers.0.mlp.down_proj"
    for side in ("su", "sv"):
        signs = tensors[f"{prefix}.{side}h"]
        packed = torch.zeros(signs.numel() // 16, dtype=torch.int16)
        for index in range(packed.numel()):
            bits = 0
            for position in range(16):
                if signs[index * 16 + position] < 0:
                    bits |= 1 << position
            packed[index] = bits if bits < 32768 else bits - 65536
        tensors[f"{prefix}.{side}"] = packed
        del tensors[f"{prefix}.{side}h"]
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader:
        actual = reader.read_operation("model.layers.0.mlp.down_proj")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exl3_rejects_incomplete_output_block(tmp_path: Path) -> None:
    _ = exl3_fixture(tmp_path, 2, "3inst")
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    prefix = "model.layers.0.mlp.down_proj"
    tensors[f"{prefix}.trellis"] = tensors[f"{prefix}.trellis"][:, :1].contiguous()
    save_file(tensors, path)
    with open_checkpoint(tmp_path) as reader, pytest.raises(ModelLoadError, match="128-column"):
        reader.read_operation("model.layers.0.mlp.down_proj")
