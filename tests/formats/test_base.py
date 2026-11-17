from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xurcroh.errors import ModelLoadError
from xurcroh.formats import open_checkpoint
from xurcroh.formats.safetensors import SafetensorsReader
from xurcroh.shards.compiler import CheckpointReader


def test_legacy_checkpoint_reader_uses_format_neutral_implementation(
    tiny_llama_path: Path,
) -> None:
    assert isinstance(CheckpointReader(tiny_llama_path), SafetensorsReader)


def test_safetensors_reader_matches_source_and_supports_bounded_columns(
    tiny_llama_path: Path,
) -> None:
    with open_checkpoint(tiny_llama_path) as reader:
        name = "model.layers.0.self_attn.q_proj.weight"
        expected = reader.read_tensors([name])[name]
        actual = reader.read_operation(name, columns=slice(1, 3))
        assert reader.source_format == "safetensors"
        assert reader.architecture == "llama"
        assert reader.encodings
        torch.testing.assert_close(actual, expected[1:3, :])
        assert len(reader.fingerprint) == 64


def test_reader_rejects_access_after_close(tiny_llama_path: Path) -> None:
    reader = open_checkpoint(tiny_llama_path)
    reader.close()
    with pytest.raises(ModelLoadError, match="closed"):
        reader.tensor_names()


def test_source_fingerprint_covers_weight_bytes(tiny_llama_path: Path, tmp_path: Path) -> None:
    cloned = tmp_path / "checkpoint"
    import shutil

    shutil.copytree(tiny_llama_path, cloned)
    with open_checkpoint(cloned) as reader:
        before = reader.fingerprint
    weight = cloned / "model.safetensors"
    payload = bytearray(weight.read_bytes())
    payload[-1] ^= 1
    weight.write_bytes(payload)
    with open_checkpoint(cloned) as reader:
        assert reader.fingerprint != before


def test_duplicate_json_tensor_names_are_rejected(tiny_llama_path: Path, tmp_path: Path) -> None:
    import shutil

    cloned = tmp_path / "checkpoint"
    shutil.copytree(tiny_llama_path, cloned)
    index = cloned / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"duplicate":"model.safetensors","duplicate":"other"}}')
    with pytest.raises(ModelLoadError, match="duplicate JSON key"):
        open_checkpoint(cloned)


def test_invalid_json_is_a_model_load_error(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text("{")
    save_file({"weight": torch.zeros((1, 1))}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ModelLoadError, match="cannot read checkpoint config"):
        open_checkpoint(tmp_path)
