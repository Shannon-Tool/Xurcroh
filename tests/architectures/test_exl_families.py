from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from tests.architectures.utils import write_family_exl3, write_llama_exl2

from xurcroh.architectures import open_architecture
from xurcroh.errors import XurcrohError
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint

# ExLlama containers keep the source ``config.json``, so the graph semantics are
# the checkpoint's own. These cases cover the layouts that differ between
# families: attention biases, query/key norms, sandwich norms and scaled
# embeddings.
FAMILIES: dict[str, dict[str, Any]] = {
    "llama": {},
    "llama-attention-bias": {"attention_bias": True},
    "mistral": {},
    "qwen2": {},
    "qwen3": {},
    "gemma3_text": {"sliding_window": 4},
}


def _model(case: str, *, width: int, heads: int, key_value_heads: int, layers: int) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type = case.split("-", 1)[0]
    head = width // heads
    config = AutoConfig.for_model(
        model_type,
        vocab_size=24,
        hidden_size=width,
        intermediate_size=width if width == 128 else 2 * width,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=key_value_heads,
        head_dim=head,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        **({"query_pre_attn_scalar": head} if model_type == "gemma3_text" else {}),
        **FAMILIES[case],
    )
    with torch.random.fork_rng():
        torch.manual_seed(417)
        model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
        for name, parameter in model.named_parameters():
            if parameter.ndim <= 1:
                parameter.data = torch.randn_like(parameter) / 4 + (
                    1.0 if "norm" in name and "gemma" not in model_type else 0.0
                )
    return model


def _assert_matches_decoded_weights(model: Any, checkpoint: Path, tmp_path: Path) -> None:
    """The container must execute exactly like its own decoded weights."""
    with open_architecture(checkpoint) as adapter:
        names = list(adapter.reader.tensor_names())
        decoded = {
            name: tensor.to(torch.float32)
            for name, tensor in adapter.reader.read_tensors(names).items()
        }
    assert set(decoded) == set(model.state_dict())
    model.load_state_dict(decoded, strict=True)
    control_path = tmp_path / "control"
    model.save_pretrained(control_path, safe_serialization=True)
    compiled = compile_fortress_checkpoint(checkpoint, seed=731, model_dtype=torch.float32)
    control = compile_fortress_checkpoint(control_path, seed=731, model_dtype=torch.float32)
    assert compiled.model is not None
    assert control.model is not None
    assert type(compiled.model) is type(model)
    assert sorted(compiled.operations) == sorted(control.operations)
    input_ids = torch.tensor([[1, 2, 3, 5, 8, 13]])
    with torch.inference_mode():
        reference = control.model(input_ids).logits
        actual = compiled.model(input_ids).logits
    # Same decoded weights and seed through the Safetensors path, so this is
    # exact up to float rounding; fixed-point quality is measured elsewhere.
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=0)


@pytest.mark.parametrize("case", sorted(FAMILIES))
def test_exl2_family_executes_like_its_decoded_weights(case: str, tmp_path: Path) -> None:
    model = _model(case, width=16, heads=4, key_value_heads=2, layers=2)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    checkpoint = write_llama_exl2(source, tmp_path / "exl2")
    with open_architecture(checkpoint) as adapter:
        assert adapter.reader.source_format == "exl2"
    _assert_matches_decoded_weights(model, checkpoint, tmp_path)


@pytest.mark.parametrize("case", sorted(FAMILIES))
def test_exl3_family_executes_like_its_decoded_weights(case: str, tmp_path: Path) -> None:
    model = _model(case, width=128, heads=1, key_value_heads=1, layers=1)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    checkpoint = write_family_exl3(source, tmp_path / "exl3")
    with open_architecture(checkpoint) as adapter:
        assert adapter.reader.source_format == "exl3"
    _assert_matches_decoded_weights(model, checkpoint, tmp_path)


def test_exl2_checkpoint_missing_a_required_bias_is_rejected(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = _model("qwen2", width=16, heads=4, key_value_heads=2, layers=1)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    checkpoint = write_llama_exl2(source, tmp_path / "exl2")
    tensors = load_file(checkpoint / "model.safetensors")
    del tensors["model.layers.0.self_attn.q_proj.bias"]
    save_file(tensors, str(checkpoint / "model.safetensors"))
    with pytest.raises(XurcrohError, match="missing required bias"):
        compile_fortress_checkpoint(checkpoint, seed=731, model_dtype=torch.float32)


def test_exl2_checkpoint_with_undefined_parameters_is_rejected(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = _model("llama", width=16, heads=4, key_value_heads=2, layers=1)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    checkpoint = write_llama_exl2(source, tmp_path / "exl2")
    tensors = load_file(checkpoint / "model.safetensors")
    tensors["model.layers.0.self_attn.q_proj.bias"] = torch.ones(16)
    save_file(tensors, str(checkpoint / "model.safetensors"))
    with pytest.raises(XurcrohError, match="does not define"):
        compile_fortress_checkpoint(checkpoint, seed=731, model_dtype=torch.float32)


def test_exl2_unknown_model_type_fails_closed(tmp_path: Path) -> None:
    model = _model("llama", width=16, heads=4, key_value_heads=2, layers=1)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    checkpoint = write_llama_exl2(source, tmp_path / "exl2")
    config = json.loads((checkpoint / "config.json").read_text())
    config["model_type"] = "future-graph"
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(XurcrohError, match="no Fortress text decoder"):
        compile_fortress_checkpoint(checkpoint, seed=731, model_dtype=torch.float32)
