from __future__ import annotations

import json
from pathlib import Path

import pytest

from xurcroh.models.catalog import ModelSpec, read_checkpoint_spec


def test_qwen38_checkpoint_uses_actual_qwen35_fields(tmp_path: Path) -> None:
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "num_hidden_layers": 64,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rope_parameters": {"mrope_interleaved": True},
        },
        "vision_config": {"deepstack_visual_indexes": []},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    spec = read_checkpoint_spec(tmp_path, model_id="Qwen/Qwen3.8-27B", revision="main")
    assert spec.hidden_size == 5120
    assert spec.query_heads == 24
    assert spec.head_dimension == 256
    assert spec.supports_delta_net
    assert spec.supports_interleaved_mrope
    assert not spec.deepstack_indexes


def test_strict_model_revision_is_required() -> None:
    with pytest.raises(ValueError, match="revision"):
        ModelSpec.from_config("Qwen/Qwen3.8-27B", {}, revision=None)
