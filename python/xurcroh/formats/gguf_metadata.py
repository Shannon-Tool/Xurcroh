from __future__ import annotations

from typing import Any

from xurcroh.errors import ModelLoadError


def _required(metadata: dict[str, Any], key: str) -> Any:
    if key not in metadata:
        raise ModelLoadError(f"GGUF checkpoint is missing required metadata: {key}")
    return metadata[key]


def gguf_to_transformers_config(metadata: dict[str, Any]) -> dict[str, Any]:
    architecture = str(_required(metadata, "general.architecture"))
    prefix = f"{architecture}."
    heads = int(_required(metadata, prefix + "attention.head_count"))
    hidden = int(_required(metadata, prefix + "embedding_length"))
    config: dict[str, Any] = {
        "model_type": architecture,
        "hidden_size": hidden,
        "num_hidden_layers": int(_required(metadata, prefix + "block_count")),
        "intermediate_size": int(_required(metadata, prefix + "feed_forward_length")),
        "max_position_embeddings": int(_required(metadata, prefix + "context_length")),
        "num_attention_heads": heads,
        "num_key_value_heads": int(metadata.get(prefix + "attention.head_count_kv", heads)),
        "head_dim": int(metadata.get(prefix + "attention.key_length", hidden // heads)),
        "rope_theta": float(metadata.get(prefix + "rope.freq_base", 10000.0)),
        "vocab_size": int(metadata.get(prefix + "vocab_size", 0)) or None,
    }
    if hidden % heads != 0 and prefix + "attention.key_length" not in metadata:
        raise ModelLoadError("GGUF hidden size is not divisible by its attention head count")
    rms = metadata.get(prefix + "attention.layer_norm_rms_epsilon")
    layer = metadata.get(prefix + "attention.layer_norm_epsilon")
    if rms is not None:
        config["rms_norm_eps"] = float(rms)
    if layer is not None:
        config["layer_norm_eps"] = float(layer)
    rope_dimensions = metadata.get(prefix + "rope.dimension_count")
    if rope_dimensions is not None:
        config["rope_dimension_count"] = int(rope_dimensions)
    if prefix + "expert_count" in metadata:
        config["num_experts"] = int(metadata[prefix + "expert_count"])
        config["num_experts_per_tok"] = int(metadata.get(prefix + "expert_used_count", 1))
    if prefix + "use_parallel_residual" in metadata:
        config["use_parallel_residual"] = bool(metadata[prefix + "use_parallel_residual"])
    scaling_type = metadata.get(prefix + "rope.scaling.type")
    if scaling_type is not None and scaling_type != "none":
        config["rope_scaling"] = {
            "type": scaling_type,
            "factor": float(_required(metadata, prefix + "rope.scaling.factor")),
            "original_max_position_embeddings": int(
                _required(metadata, prefix + "rope.scaling.original_context_length")
            ),
        }
    if config["vocab_size"] is None:
        config.pop("vocab_size")
    return config
