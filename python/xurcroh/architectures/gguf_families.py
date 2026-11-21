"""Verified llama.cpp GGUF families.

A GGUF tensor name does not carry the semantics of the graph that consumes it:
normalisation, activation, rotary layout and stored-value conventions all come
from the ``general.architecture`` string. Each entry below records the tensor
naming, stored-value conventions and Transformers configuration of one family,
and exists only because an official export of that family was compared tensor
for tensor against its source checkpoint. Architectures without an entry are
rejected rather than guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from xurcroh.errors import CapabilityError, ModelLoadError

ConfigBuilder = Callable[[dict[str, Any], str, dict[str, Any], frozenset[str]], dict[str, Any]]

_SIMPLE = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_ATTENTION = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
}
_ATTENTION_BIAS = {
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.bias": "self_attn.o_proj.bias",
}
_QK_NORM = {
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}
_DENSE_MLP = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_ROUTER = {"ffn_gate_inp.weight": "mlp.gate.weight"}
_PRE_NORM = {"ffn_norm.weight": "post_attention_layernorm.weight"}
_SANDWICH_NORM = {
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
}
# Early llama.cpp exports of the Llama family used the original checkpoint names.
_LEGACY_SIMPLE = {"tok_embeddings.weight": "model.embed_tokens.weight"}
_LEGACY_LAYER = {
    "attention_norm.weight": "input_layernorm.weight",
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}

EXPERT_STACKS = {
    "ffn_gate_exps.weight": "gate_proj",
    "ffn_up_exps.weight": "up_proj",
    "ffn_down_exps.weight": "down_proj",
}
FUSED_QKV = "attn_qkv.weight"

# Layouts that are recognised but deliberately not executed. Anything matching
# neither a family table nor an entry here is reported as unrecognised.
UNSUPPORTED_MARKERS = (
    (
        "position_embd",
        "learned absolute positional embeddings are outside the Fortress rotary boundary",
    ),
    ("token_types", "token-type embeddings are not supported"),
    ("attn_norm_2", "parallel dual-residual attention layouts are not supported"),
    ("attn_q_a", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("attn_q_b", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("attn_kv_a", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("attn_kv_b", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("attn_k_b", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("attn_v_b", "DeepSeek MLA GGUF layouts are not supported; use the safetensors checkpoint"),
    ("shexp", "shared-expert GGUF layouts are not supported; use the safetensors checkpoint"),
)
# Tensor-name prefixes of client-local media towers; they are not part of the
# protected text graph and are neither mapped nor rejected.
MEDIA_PREFIXES = ("v.", "a.", "mm.", "vision.", "audio.")


def _tied(names: frozenset[str]) -> bool:
    return "output.weight" not in names


def _has_bias(names: frozenset[str], suffix: str) -> bool:
    return any(name.startswith("blk.") and name.endswith(suffix) for name in names)


def _required(metadata: dict[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ModelLoadError(f"GGUF checkpoint is missing required metadata: {key}")
    return metadata[key]


def _llama(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    return {
        **base,
        "tie_word_embeddings": _tied(names),
        "attention_bias": _has_bias(names, ".attn_q.bias"),
        "mlp_bias": False,
    }


def _qwen2(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    if not _has_bias(names, ".attn_q.bias"):
        raise CapabilityError("qwen2 GGUF checkpoints must carry query/key/value biases")
    return {**base, "tie_word_embeddings": _tied(names)}


def _qwen3(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    return {**base, "tie_word_embeddings": _tied(names), "attention_bias": False}


def _qwen3_moe(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    return {
        **_qwen3(metadata, prefix, base, names),
        "moe_intermediate_size": int(_required(metadata, prefix + "expert_feed_forward_length")),
        "num_experts": int(_required(metadata, prefix + "expert_count")),
        "num_experts_per_tok": int(_required(metadata, prefix + "expert_used_count")),
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        # llama.cpp renormalises the selected expert weights for this family.
        "norm_topk_prob": True,
    }


def _gemma3(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    layers = int(base["num_hidden_layers"])
    # llama.cpp hard-codes the Gemma 3 pattern: every sixth layer attends globally.
    layer_types = [
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(layers)
    ]
    config = {
        **base,
        "tie_word_embeddings": _tied(names),
        "sliding_window": int(_required(metadata, prefix + "attention.sliding_window")),
        "layer_types": layer_types,
        "query_pre_attn_scalar": int(base["head_dim"]),
        "rope_local_base_freq": 10000.0,
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
    }
    return config


def _gemma4(
    metadata: dict[str, Any], prefix: str, base: dict[str, Any], names: frozenset[str]
) -> dict[str, Any]:
    layers = int(base["num_hidden_layers"])
    pattern = _required(metadata, prefix + "attention.sliding_window_pattern")
    widths = _required(metadata, prefix + "feed_forward_length")
    if not isinstance(pattern, list) or len(pattern) != layers:
        raise ModelLoadError("gemma4 sliding-window pattern does not match the layer count")
    if not isinstance(widths, list) or len(widths) != layers:
        raise ModelLoadError("gemma4 feed-forward lengths do not match the layer count")
    shared = int(metadata.get(prefix + "attention.shared_kv_layers", 0))
    narrow = int(widths[0])
    wide = {int(width) for width in widths[layers - shared :]} if shared else set()
    if any(int(width) != narrow for width in widths[: layers - shared]) or wide - {
        narrow,
        2 * narrow,
    }:
        raise CapabilityError("gemma4 feed-forward layout is not a supported double-wide pattern")
    global_head = int(_required(metadata, prefix + "attention.key_length"))
    local_head = int(_required(metadata, prefix + "attention.key_length_swa"))
    softcap = metadata.get(prefix + "final_logit_softcapping")
    per_layer = int(_required(metadata, prefix + "embedding_length_per_layer_input"))
    vocabulary = base.get("vocab_size")
    config = {
        key: value
        for key, value in base.items()
        if key not in {"rope_theta", "rope_dimension_count"}
    }
    config.update(
        {
            "tie_word_embeddings": _tied(names),
            "intermediate_size": narrow,
            "use_double_wide_mlp": wide == {2 * narrow},
            "head_dim": local_head,
            "global_head_dim": global_head,
            "sliding_window": int(_required(metadata, prefix + "attention.sliding_window")),
            "layer_types": [
                "sliding_attention" if bool(sliding) else "full_attention" for sliding in pattern
            ],
            "num_kv_shared_layers": shared,
            "hidden_size_per_layer_input": per_layer,
            "final_logit_softcapping": None if softcap is None else float(softcap),
            "rope_parameters": {
                "full_attention": {
                    # Fixed by the Gemma 4 graph; not recorded in GGUF metadata.
                    "partial_rotary_factor": 0.25,
                    "rope_theta": float(_required(metadata, prefix + "rope.freq_base")),
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": float(_required(metadata, prefix + "rope.freq_base_swa")),
                    "rope_type": "default",
                },
            },
            "enable_moe_block": False,
        }
    )
    if vocabulary is not None:
        config["vocab_size_per_layer_input"] = vocabulary
    return config


@dataclass(frozen=True)
class GGUFFamily:
    architecture: str
    model_type: str
    layer: Mapping[str, str]
    config: ConfigBuilder
    simple: Mapping[str, str] = field(default_factory=lambda: dict(_SIMPLE))
    derived: frozenset[str] = frozenset()
    norm_offset: float = 0.0
    permute_query_key: bool = False
    experts: bool = False
    fused_qkv: bool = False


_LLAMA_LAYER = {**_ATTENTION, **_ATTENTION_BIAS, **_DENSE_MLP, **_ROUTER, **_PRE_NORM}

FAMILIES: dict[str, GGUFFamily] = {
    family.architecture: family
    for family in (
        GGUFFamily(
            "llama",
            "llama",
            {**_LLAMA_LAYER, **_LEGACY_LAYER},
            _llama,
            simple={**_SIMPLE, **_LEGACY_SIMPLE},
            # llama.cpp stores query/key rows for interleaved-pair rotation; the
            # Transformers Llama graph rotates split halves.
            permute_query_key=True,
            experts=True,
            fused_qkv=True,
        ),
        GGUFFamily(
            "qwen2", "qwen2", {**_ATTENTION, **_ATTENTION_BIAS, **_DENSE_MLP, **_PRE_NORM}, _qwen2
        ),
        GGUFFamily("qwen3", "qwen3", {**_ATTENTION, **_QK_NORM, **_DENSE_MLP, **_PRE_NORM}, _qwen3),
        GGUFFamily(
            "qwen3moe",
            "qwen3_moe",
            {**_ATTENTION, **_QK_NORM, **_ROUTER, **_PRE_NORM},
            _qwen3_moe,
            experts=True,
        ),
        GGUFFamily(
            "gemma3",
            "gemma3_text",
            {**_ATTENTION, **_QK_NORM, **_DENSE_MLP, **_SANDWICH_NORM},
            _gemma3,
            # The llama.cpp converter adds one to every Gemma 3 norm weight.
            norm_offset=1.0,
        ),
        GGUFFamily(
            "gemma4",
            "gemma4_text",
            {
                **_ATTENTION,
                **_QK_NORM,
                **_DENSE_MLP,
                **_SANDWICH_NORM,
                "layer_output_scale.weight": "layer_scalar",
                "inp_gate.weight": "per_layer_input_gate.weight",
                "proj.weight": "per_layer_projection.weight",
                "post_norm.weight": "post_per_layer_input_norm.weight",
            },
            _gemma4,
            simple={
                **_SIMPLE,
                "per_layer_token_embd.weight": "model.embed_tokens_per_layer.weight",
                "per_layer_model_proj.weight": "model.per_layer_model_projection.weight",
                "per_layer_proj_norm.weight": "model.per_layer_projection_norm.weight",
            },
            derived=frozenset({"rope_freqs.weight"}),
        ),
    )
}


def gguf_family(architecture: str) -> GGUFFamily:
    family = FAMILIES.get(architecture)
    if family is None:
        raise CapabilityError(
            f"GGUF architecture {architecture!r} has no verified Fortress semantic adapter; "
            f"supported architectures: {', '.join(sorted(FAMILIES))}"
        )
    return family


def gguf_model_type(architecture: str) -> str | None:
    family = FAMILIES.get(architecture)
    return None if family is None else family.model_type
