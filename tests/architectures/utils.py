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
