from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open
from tests.formats.gguf_utils import write_gguf


def _gguf_name(name: str) -> str:
    replacements = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    if name in replacements:
        return replacements[name]
    if name.startswith("model.layers."):
        suffix = name.removeprefix("model.layers.")
        layer, remainder = suffix.split(".", 1)
        attention = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if remainder in attention:
            return f"blk.{layer}.{attention[remainder]}"
    raise ValueError(f"no GGUF mapping for {name}")


def write_llama_gguf(source: Path, destination: Path) -> Path:
    config = json.loads((source / "config.json").read_text())
    tensors = []
    with safe_open(source / "model.safetensors", framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            tensor = handle.get_tensor(name)
            tensors.append((_gguf_name(name), tuple(tensor.shape), 0, tensor.numpy().tobytes()))
    metadata = {
        "llama.embedding_length": config["hidden_size"],
        "llama.block_count": config["num_hidden_layers"],
        "llama.feed_forward_length": config["intermediate_size"],
        "llama.context_length": config["max_position_embeddings"],
        "llama.attention.head_count": config["num_attention_heads"],
        "llama.attention.head_count_kv": config["num_key_value_heads"],
        "llama.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "llama.rope.freq_base": config.get("rope_theta", 10000.0),
        "llama.vocab_size": config["vocab_size"],
    }
    tokenizer_path = source / "tokenizer.json"
    if tokenizer_path.is_file():
        metadata["tokenizer.huggingface.json"] = tokenizer_path.read_text()
        tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
        if isinstance(tokenizer_config.get("chat_template"), str):
            metadata["tokenizer.chat_template"] = tokenizer_config["chat_template"]
        if tokenizer_config.get("eos_token_id") is not None:
            metadata["tokenizer.ggml.eos_token_id"] = tokenizer_config["eos_token_id"]
    return write_gguf(destination, tensors, metadata=metadata)


def write_llama_exl2(source: Path, destination: Path) -> Path:
    import shutil

    import torch
    from safetensors.torch import save_file
    from tests.formats.exl_utils import pack_columns, pack_scales

    destination.mkdir(parents=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    output = {}
    with safe_open(source / "model.safetensors", framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            weight = handle.get_tensor(name)
            if weight.ndim != 2 or not any(
                marker in name
                for marker in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )
            ):
                output[name] = weight
                continue
            prefix = name.removesuffix(".weight")
            matrix = weight.T
            input_size, output_size = matrix.shape
            padded = ((output_size + 31) // 32) * 32
            groups = []
            quantized_groups = []
            packed_row = 0
            for start in range(0, input_size, 32):
                rows = min(32, input_size - start)
                scale = torch.tensor(0.01, dtype=torch.float16)
                quantized = torch.clamp(
                    torch.round(matrix[start : start + rows] / scale) + 8, 0, 15
                ).to(torch.int16)
                quantized_groups.append(quantized)
                groups.extend((4, packed_row))
                packed_row += (rows * 4 + 31) // 32
            quantized = torch.cat(quantized_groups)
            padded_quantized = torch.full((input_size, padded), 8, dtype=torch.int16)
            padded_quantized[:, :output_size] = quantized
            output[f"{prefix}.q_weight"] = pack_columns(padded_quantized, 4)
            output[f"{prefix}.q_scale"] = pack_scales(
                torch.ones((len(groups) // 2, padded), dtype=torch.int16)
            )
            output[f"{prefix}.q_scale_max"] = torch.full(
                (len(groups) // 2,), 0.01, dtype=torch.float16
            )
            output[f"{prefix}.q_groups"] = torch.tensor(groups, dtype=torch.int16)
            output[f"{prefix}.q_invperm"] = torch.arange(input_size, dtype=torch.int32)
    save_file(output, str(destination / "model.safetensors"))
    return destination


def write_llama_exl3(source: Path, destination: Path) -> Path:
    import shutil

    import torch
    from safetensors.torch import load_file, save_file
    from tests.formats.exl3_utils import exl3_fixture

    destination.mkdir(parents=True)
    exl3_fixture(destination, 2, "3inst")
    encoded = load_file(destination / "model.safetensors")
    output = {}
    operations = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    )
    source_prefix = "model.layers.0.mlp.down_proj"
    for prefix in operations:
        for suffix in ("trellis", "suh", "svh"):
            output[f"{prefix}.{suffix}"] = encoded[f"{source_prefix}.{suffix}"].clone()
    config = json.loads((source / "config.json").read_text())
    config.update(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=128,
    )
    (destination / "config.json").write_text(json.dumps(config))
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    generator = torch.Generator().manual_seed(91)
    output["model.embed_tokens.weight"] = torch.randn(
        config["vocab_size"], 128, generator=generator
    )
    output["lm_head.weight"] = torch.randn(config["vocab_size"], 128, generator=generator)
    output["model.norm.weight"] = torch.ones(128)
    output["model.layers.0.input_layernorm.weight"] = torch.ones(128)
    output["model.layers.0.post_attention_layernorm.weight"] = torch.ones(128)
    save_file(output, str(destination / "model.safetensors"))
    return destination


_FAMILY_LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.bias": "attn_output.bias",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
}
_PRE_NORM_NAMES = {"post_attention_layernorm.weight": "ffn_norm.weight"}
_SANDWICH_NORM_NAMES = {
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "pre_feedforward_layernorm.weight": "ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
}
_GEMMA4_LAYER_NAMES = {
    "layer_scalar": "layer_output_scale.weight",
    "per_layer_input_gate.weight": "inp_gate.weight",
    "per_layer_projection.weight": "proj.weight",
    "post_per_layer_input_norm.weight": "post_norm.weight",
}
_FAMILY_SIMPLE_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "model.embed_tokens_per_layer.weight": "per_layer_token_embd.weight",
    "model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
    "model.per_layer_projection_norm.weight": "per_layer_proj_norm.weight",
}


def _llama_cpp_rows(tensor, heads: int):  # type: ignore[no-untyped-def]
    """Reorder split-half rotary rows into llama.cpp's interleaved-pair order."""
    rows = tensor.shape[0]
    return (
        tensor.reshape(heads, 2, rows // heads // 2, *tensor.shape[1:])
        .swapaxes(1, 2)
        .reshape(tensor.shape)
    )


class _PlainConfig:
    """Reads config values, including per-layer ones that heterogeneous configs guard."""

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        self._config = config
        self._values = config.to_dict()

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        values = self.__dict__["_values"]
        if name in values:
            return values[name]
        return getattr(self.__dict__["_config"], name)


def _key_length(state, config, kind: str) -> int:  # type: ignore[no-untyped-def]
    # Per-layer head widths are read from the weights of a layer of that kind.
    layer = list(config.layer_types).index(kind)
    rows = state[f"model.layers.{layer}.self_attn.q_proj.weight"].shape[0]
    return int(rows // int(config.num_attention_heads))


def _rope_theta(config) -> float:  # type: ignore[no-untyped-def]
    parameters = getattr(config, "rope_parameters", None) or {}
    if "rope_theta" in parameters:
        return float(parameters["rope_theta"])
    if "full_attention" in parameters:
        return float(parameters["full_attention"]["rope_theta"])
    return float(config.rope_theta)


def write_family_gguf(model, destination: Path, architecture: str) -> Path:  # type: ignore[no-untyped-def]
    """Export a Transformers decoder the way llama.cpp's converter does.

    Applies the conventions observed in official exports: per-family tensor
    names, Llama query/key row order, Gemma 3 norm offset, stacked experts and
    omitted tied output heads.
    """
    import torch

    config = _PlainConfig(model.config)
    prefix = f"{architecture}."
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    sandwich = architecture in {"gemma3", "gemma4"}
    layer_names = {
        **_FAMILY_LAYER_NAMES,
        **(_SANDWICH_NORM_NAMES if sandwich else _PRE_NORM_NAMES),
        **(_GEMMA4_LAYER_NAMES if architecture == "gemma4" else {}),
    }
    state = {name: tensor.detach().to(torch.float32) for name, tensor in model.state_dict().items()}
    tied = bool(getattr(config, "tie_word_embeddings", False))
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def add(name: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        shape = tuple(tensor.shape) or (1,)
        tensors.append((name, shape, 0, tensor.numpy().tobytes()))

    for name, tensor in state.items():
        if name in _FAMILY_SIMPLE_NAMES:
            if name == "lm_head.weight" and tied:
                continue
            if architecture == "gemma3" and "norm" in name:
                tensor = tensor + 1
            add(_FAMILY_SIMPLE_NAMES[name], tensor)
            continue
        if not name.startswith("model.layers."):
            raise ValueError(f"no GGUF mapping for {name}")
        layer, remainder = name.removeprefix("model.layers.").split(".", 1)
        if remainder == "mlp.experts.gate_up_proj":
            gate, up = tensor.chunk(2, dim=1)
            add(f"blk.{layer}.ffn_gate_exps.weight", gate)
            add(f"blk.{layer}.ffn_up_exps.weight", up)
            continue
        if remainder == "mlp.experts.down_proj":
            add(f"blk.{layer}.ffn_down_exps.weight", tensor)
            continue
        if remainder not in layer_names:
            raise ValueError(f"no GGUF mapping for {name}")
        if architecture == "llama" and remainder.startswith("self_attn.q_proj"):
            tensor = _llama_cpp_rows(tensor, heads)
        if architecture == "llama" and remainder.startswith("self_attn.k_proj"):
            tensor = _llama_cpp_rows(tensor, kv_heads)
        if architecture == "gemma3" and "norm" in remainder:
            tensor = tensor + 1
        add(f"blk.{layer}.{layer_names[remainder]}", tensor)

    metadata: dict[str, object] = {
        "general.architecture": architecture,
        prefix + "embedding_length": int(config.hidden_size),
        prefix + "block_count": int(config.num_hidden_layers),
        prefix + "feed_forward_length": int(config.intermediate_size),
        prefix + "context_length": int(config.max_position_embeddings),
        prefix + "attention.head_count": heads,
        prefix + "attention.head_count_kv": kv_heads,
        prefix + "attention.layer_norm_rms_epsilon": float(config.rms_norm_eps),
    }
    head_dim = None if architecture == "gemma4" else getattr(config, "head_dim", None)
    if head_dim is not None:
        metadata[prefix + "attention.key_length"] = int(head_dim)
    if architecture in {"llama", "qwen2", "qwen3", "qwen3moe"}:
        metadata[prefix + "rope.freq_base"] = _rope_theta(config)
        metadata[prefix + "vocab_size"] = int(config.vocab_size)
    if architecture == "qwen3moe":
        metadata[prefix + "expert_count"] = int(config.num_experts)
        metadata[prefix + "expert_used_count"] = int(config.num_experts_per_tok)
        metadata[prefix + "expert_feed_forward_length"] = int(config.moe_intermediate_size)
    if architecture == "gemma3":
        metadata[prefix + "rope.freq_base"] = _rope_theta(config)
        metadata[prefix + "attention.sliding_window"] = int(config.sliding_window)
    if architecture == "gemma4":
        rope = config.rope_parameters
        wide = 2 if config.use_double_wide_mlp else 1
        shared = int(config.num_kv_shared_layers)
        layers = int(config.num_hidden_layers)
        metadata.update(
            {
                prefix + "feed_forward_length": [int(config.intermediate_size)] * (layers - shared)
                + [int(config.intermediate_size) * wide] * shared,
                prefix + "attention.key_length": _key_length(state, config, "full_attention"),
                prefix + "attention.key_length_swa": _key_length(
                    state, config, "sliding_attention"
                ),
                prefix + "attention.sliding_window": int(config.sliding_window),
                prefix + "attention.sliding_window_pattern": [
                    kind == "sliding_attention" for kind in config.layer_types
                ],
                prefix + "attention.shared_kv_layers": shared,
                prefix + "embedding_length_per_layer_input": int(
                    config.hidden_size_per_layer_input
                ),
                prefix + "rope.freq_base": float(rope["full_attention"]["rope_theta"]),
                prefix + "rope.freq_base_swa": float(rope["sliding_attention"]["rope_theta"]),
            }
        )
        if config.final_logit_softcapping is not None:
            metadata[prefix + "final_logit_softcapping"] = float(config.final_logit_softcapping)
    return write_gguf(destination, tensors, metadata=metadata)


def write_family_exl3(source: Path, destination: Path) -> Path:
    """Re-encode a saved 128-wide single-layer decoder as EXL3.

    Every protected projection takes the same verified trellis fixture; all
    other tensors (norms, biases, embeddings) are copied from ``source``.
    """
    import shutil

    from safetensors.torch import load_file, save_file
    from tests.formats.exl3_utils import exl3_fixture

    scratch = destination.parent / f"{destination.name}-trellis"
    scratch.mkdir(parents=True)
    exl3_fixture(scratch, 2, "3inst")
    encoded = load_file(scratch / "model.safetensors")
    shutil.rmtree(scratch)
    destination.mkdir(parents=True)
    output = {}
    for name, tensor in load_file(source / "model.safetensors").items():
        prefix = name.removesuffix(".weight")
        if tensor.ndim == 2 and prefix.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        ):
            if tuple(tensor.shape) != (128, 128):
                raise ValueError(f"EXL3 fixture needs 128x128 projections: {name}")
            for suffix in ("trellis", "suh", "svh"):
                output[f"{prefix}.{suffix}"] = encoded[
                    f"model.layers.0.mlp.down_proj.{suffix}"
                ].clone()
        else:
            output[name] = tensor
    save_file(output, str(destination / "model.safetensors"))
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    return destination
