from __future__ import annotations

from pathlib import Path

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from Xurcroh import Xurcroh


def copy_tokenizer(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if path.name.startswith("tokenizer") or path.name in {"vocab.json", "merges.txt"}:
            (destination / path.name).write_bytes(path.read_bytes())


def test_official_gemma4_shared_kv_and_per_layer_features(
    tmp_path: Path, tiny_llama_path: Path
) -> None:
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        hidden_size_per_layer_input=4,
        num_kv_shared_layers=1,
        use_double_wide_mlp=True,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
            },
        },
    )
    model = Gemma4ForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    copy_tokenizer(tiny_llama_path, tmp_path)
    (tmp_path / "LICENSE").write_text("Apache-2.0")
    with Xurcroh(model_path={"gemma": {"path": str(tmp_path), "engine": "fortress"}}) as client:
        response = client.chat.completions.create(
            model="gemma",
            messages=[{"role": "user", "content": "1"}],
            temperature=0,
            max_tokens=1,
        )
    assert response.usage.completion_tokens == 1
    assert torch.isfinite(torch.tensor(float(response.execution.duration_ms)))


def test_official_qwen35_gated_delta_net(tmp_path: Path, tiny_llama_path: Path) -> None:
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    model = Qwen3_5ForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    copy_tokenizer(tiny_llama_path, tmp_path)
    with Xurcroh(model_path={"qwen": {"path": str(tmp_path), "engine": "fortress"}}) as client:
        response = client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": "1"}],
            temperature=0,
            max_tokens=1,
        )
    assert response.usage.completion_tokens == 1
    assert response.execution.engine == "fortress-local"
