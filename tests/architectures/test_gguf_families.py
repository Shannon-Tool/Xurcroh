from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tests.architectures.utils import write_family_gguf
from tests.formats.gguf_utils import write_gguf

from xurcroh.architectures import open_architecture
from xurcroh.errors import CapabilityError
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint


def _llama(**overrides: Any) -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **{"tie_word_embeddings": False, **overrides},
        )
    )


def _qwen2() -> Any:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    )


def _qwen3() -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


def _qwen3_moe() -> Any:
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    )


def _gemma3() -> Any:
    from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3TextConfig

    return Gemma3ForCausalLM(
        Gemma3TextConfig(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            query_pre_attn_scalar=8,
            sliding_window=4,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


def _gemma4() -> Any:
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

    return Gemma4ForCausalLM(
        Gemma4TextConfig(
            vocab_size=24,
            vocab_size_per_layer_input=24,
            hidden_size=16,
            hidden_size_per_layer_input=4,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=4,
            global_head_dim=8,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            sliding_window=4,
            max_position_embeddings=64,
            tie_word_embeddings=True,
        )
    )


FAMILIES = {
    "llama": ("llama", _llama),
    "llama-tied": ("llama", lambda: _llama(tie_word_embeddings=True)),
    "llama-attention-bias": ("llama", lambda: _llama(attention_bias=True)),
    "qwen2": ("qwen2", _qwen2),
    "qwen3": ("qwen3", _qwen3),
    "qwen3moe": ("qwen3moe", _qwen3_moe),
    "gemma3": ("gemma3", _gemma3),
    "gemma4": ("gemma4", _gemma4),
}


def _build(case: str, tmp_path: Path) -> tuple[Any, Path]:
    architecture, factory = FAMILIES[case]
    with torch.random.fork_rng():
        torch.manual_seed(731)
        model = factory().to(torch.float32).eval()
        # Random norm gains and biases make a dropped or mis-mapped tensor visible.
        for name, parameter in model.named_parameters():
            if parameter.ndim <= 1:
                parameter.data = torch.randn_like(parameter) / 4 + (
                    1.0 if "norm" in name and "gemma" not in architecture else 0.0
                )
    return model, write_family_gguf(model, tmp_path / f"{case}.gguf", architecture)


@pytest.mark.parametrize("case", sorted(FAMILIES))
def test_gguf_family_tensors_round_trip_to_canonical_names(case: str, tmp_path: Path) -> None:
    model, checkpoint = _build(case, tmp_path)
    state = model.state_dict()
    with open_architecture(checkpoint) as adapter:
        names = adapter.reader.tensor_names()
        decoded = adapter.reader.read_tensors(list(names))
        assert adapter.reader.config["tie_word_embeddings"] == model.config.tie_word_embeddings
    assert ("lm_head.weight" in names) == (not model.config.tie_word_embeddings)
    for name, tensor in decoded.items():
        if ".experts." in name:
            continue
        torch.testing.assert_close(tensor.float(), state[name].float(), atol=1e-6, rtol=0)
    expected = {
        name
        for name in state
        if ".experts." not in name
        and not (name == "lm_head.weight" and "lm_head.weight" not in names)
    }
    assert expected <= set(names)


@pytest.mark.parametrize("case", sorted(FAMILIES))
def test_gguf_family_executes_like_the_source_model(case: str, tmp_path: Path) -> None:
    model, checkpoint = _build(case, tmp_path)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    compiled = compile_fortress_checkpoint(checkpoint, seed=731, model_dtype=torch.float32)
    control = compile_fortress_checkpoint(source, seed=731, model_dtype=torch.float32)
    assert compiled.model is not None
    assert control.model is not None
    assert type(compiled.model) is type(model)
    assert sorted(compiled.operations) == sorted(control.operations)
    input_ids = torch.tensor([[1, 2, 3, 5, 8, 13]])
    with torch.inference_mode():
        expected = model(input_ids).logits
        reference = control.model(input_ids).logits
        actual = compiled.model(input_ids).logits
    # Same weights and seed through the Safetensors path: any difference is a
    # GGUF naming, ordering or configuration error rather than quantization.
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=0)
    assert float((actual - expected).abs().max()) < 0.25 * float(expected.abs().max())


def _tensor(name: str, *shape: int) -> tuple[str, tuple[int, ...], int, bytes]:
    return (name, shape, 0, np.zeros(shape, dtype=np.float32).tobytes())


def _minimal(*extra: tuple[str, tuple[int, ...], int, bytes]) -> list[Any]:
    return [_tensor("token_embd.weight", 4, 4), _tensor("blk.0.attn_q.weight", 4, 4), *extra]


def test_unverified_gguf_architecture_fails_closed(tmp_path: Path) -> None:
    metadata = {
        "general.architecture": "phi3",
        "phi3.embedding_length": 4,
        "phi3.block_count": 1,
        "phi3.feed_forward_length": 8,
        "phi3.context_length": 16,
        "phi3.attention.head_count": 2,
    }
    checkpoint = write_gguf(tmp_path / "phi3.gguf", _minimal(), metadata=metadata)
    with pytest.raises(CapabilityError, match="'phi3' has no verified Fortress semantic adapter"):
        open_architecture(checkpoint)


@pytest.mark.parametrize(
    ("tensor", "reason"),
    [
        (_tensor("position_embd.weight", 16, 4), "learned absolute positional embeddings"),
        (_tensor("blk.0.attn_norm_2.weight", 4), "parallel dual-residual"),
        (_tensor("blk.0.attn_kv_b.weight", 4, 4), "DeepSeek MLA"),
        (_tensor("blk.0.ffn_gate_shexp.weight", 4, 4), "shared-expert"),
        (_tensor("blk.0.ffn_up.bias", 8), "bias tensors"),
        (_tensor("output.bias", 4), "bias tensors"),
        (_tensor("blk.0.mystery.weight", 4, 4), "unrecognized GGUF tensor"),
        (_tensor("mystery.weight", 4, 4), "unrecognized GGUF tensor"),
    ],
)
def test_gguf_tensors_outside_the_family_are_reported(
    tensor: tuple[str, tuple[int, ...], int, bytes], reason: str, tmp_path: Path
) -> None:
    checkpoint = write_gguf(tmp_path / "extra.gguf", _minimal(tensor))
    with pytest.raises(CapabilityError, match=reason):
        open_architecture(checkpoint)


def test_family_specific_tensor_is_rejected_in_another_family(tmp_path: Path) -> None:
    # Gemma's sandwich norms mean something else in a Llama graph.
    checkpoint = write_gguf(
        tmp_path / "mixed.gguf", _minimal(_tensor("blk.0.post_ffw_norm.weight", 4))
    )
    with pytest.raises(CapabilityError, match="post_ffw_norm"):
        open_architecture(checkpoint)


def test_gguf_without_embedding_or_blocks_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CapabilityError, match="token embedding"):
        open_architecture(write_gguf(tmp_path / "a.gguf", [_tensor("blk.0.attn_q.weight", 4, 4)]))
    with pytest.raises(CapabilityError, match="no transformer blocks"):
        open_architecture(write_gguf(tmp_path / "b.gguf", [_tensor("token_embd.weight", 4, 4)]))


def test_fused_qkv_is_split_per_layer(tmp_path: Path) -> None:
    first = np.arange(32, dtype=np.float32).reshape(8, 4)
    second = first + 100
    checkpoint = write_gguf(
        tmp_path / "fused.gguf",
        [
            _tensor("token_embd.weight", 4, 4),
            ("blk.0.attn_qkv.weight", (8, 4), 0, first.tobytes()),
            ("blk.1.attn_qkv.weight", (8, 4), 0, second.tobytes()),
        ],
        metadata={"llama.block_count": 2},
    )
    with open_architecture(checkpoint) as adapter:
        name = "model.layers.1.self_attn.v_proj.weight"
        assert adapter.reader.tensor_shapes([name])[name] == (2, 4)
        value = adapter.reader.read_tensors([name])[name]
        with pytest.raises(Exception, match="unsupported GGUF logical tensor"):
            adapter.reader.tensor_shapes(["model.layers.1.self_attn.o_proj.weight"])
    torch.testing.assert_close(value, torch.from_numpy(second[6:]))


def test_qwen2_gguf_without_attention_biases_is_rejected(tmp_path: Path) -> None:
    metadata = {
        "general.architecture": "qwen2",
        "qwen2.embedding_length": 4,
        "qwen2.block_count": 1,
        "qwen2.feed_forward_length": 8,
        "qwen2.context_length": 16,
        "qwen2.attention.head_count": 2,
    }
    checkpoint = write_gguf(tmp_path / "qwen2.gguf", _minimal(), metadata=metadata)
    with pytest.raises(CapabilityError, match="query/key/value biases"):
        open_architecture(checkpoint)


def test_large_tokenizer_metadata_arrays_are_accepted(tmp_path: Path) -> None:
    # Gemma 4 exports carry three 262,144-entry vocabulary arrays plus merges.
    metadata = {f"tokenizer.ggml.array{index}": [0] * 400_000 for index in range(3)}
    checkpoint = write_gguf(tmp_path / "vocabulary.gguf", _minimal(), metadata=metadata)
    with open_architecture(checkpoint) as adapter:
        assert "model.embed_tokens.weight" in adapter.reader.tensor_names()
