from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import ModelLoadError
from xurcroh.formats import detect_checkpoint_format, open_checkpoint
from xurcroh.preflight import ensure_checkpoint, preflight_checkpoint


def test_gguf_detection_uses_magic_not_extension(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
    assert detect_checkpoint_format(checkpoint) == "gguf"


def test_invalid_gguf_is_not_accepted_by_extension(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.gguf"
    checkpoint.write_bytes(b"not a gguf")
    with pytest.raises(ModelLoadError, match="unrecognized checkpoint"):
        detect_checkpoint_format(checkpoint)


def test_quantized_safetensors_encoding_is_detected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    save_file(
        {"model.layers.0.self_attn.q_proj.q_weight": torch.zeros((1, 1), dtype=torch.int32)},
        str(tmp_path / "model.safetensors"),
    )
    assert detect_checkpoint_format(tmp_path) == "exl2"


def test_incomplete_gguf_reader_input_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.gguf"
    checkpoint.write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
    with pytest.raises(ModelLoadError, match="truncated"):
        open_checkpoint(checkpoint)


def test_local_model_file_is_accepted_without_download(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.gguf"
    checkpoint.write_bytes(b"GGUF" + (3).to_bytes(4, "little"))
    assert ensure_checkpoint("local", ModelDefinition(path=str(checkpoint))) == checkpoint


def test_unknown_architecture_does_not_pass_preflight(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    import shutil

    cloned = tmp_path / "checkpoint"
    shutil.copytree(tiny_llama_path, cloned)
    config = json.loads((cloned / "config.json").read_text())
    config["model_type"] = "unimplemented_graph"
    (cloned / "config.json").write_text(json.dumps(config))
    report = preflight_checkpoint(
        "unknown",
        ModelDefinition(path=str(cloned), engine="fortress"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    assert report.supported is False
    assert report.source_format == "safetensors"
    assert report.architecture == "unimplemented_graph"
    assert report.required_adapter == "unimplemented_graph"
    assert any("semantic adapter" in issue for issue in report.issues)
