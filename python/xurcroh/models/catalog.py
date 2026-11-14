from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_IDS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B",
    "google/gemma-4-E2B-it",
    "Qwen/Qwen3.8-27B",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.2-dev",
    "Qwen/Qwen-Image",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "MiniMaxAI/MiniMax-H3",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    model_type: str
    architecture: str
    engine: str
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    layer_count: int
    layer_types: tuple[str, ...]
    deepstack_indexes: tuple[int, ...]
    supports_interleaved_mrope: bool

    @classmethod
    def from_config(
        cls,
        model_id: str,
        config: dict[str, Any],
        *,
        revision: str | None,
    ) -> ModelSpec:
        if not revision:
            raise ValueError("a pinned model revision is required for strict compatibility")
        model_type = config.get("model_type", "unknown")
        text = {**config, **config.get("text_config", {})}
        hidden = int(text.get("hidden_size", 0))
        query_heads = int(text.get("num_attention_heads", 0))
        kv_heads = int(text.get("num_key_value_heads", query_heads))
        head_dimension = int(text.get("head_dim", hidden // query_heads if query_heads else 0))
        model_type_lower = model_type.lower()
        if "flux" in model_type_lower or "qwen_image" in model_type_lower:
            engine = "xdit"
        elif "wan" in model_type_lower or "minimax" in model_type_lower:
            engine = "sglang-diffusion"
        elif (
            "deepseek" in model_type_lower
            or "qwen" in model_type_lower
            or "gemma" in model_type_lower
        ):
            engine = "sglang"
        else:
            engine = "vllm"
        architectures = config.get("architectures", [])
        rope = text.get("rope_parameters", {})
        deepstack = config.get("vision_config", {}).get("deepstack_visual_indexes", [])
        return cls(
            model_id=model_id,
            revision=revision,
            model_type=model_type,
            architecture=architectures[0] if architectures else "unknown",
            engine=engine,
            hidden_size=hidden,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dimension=head_dimension,
            layer_count=int(text.get("num_hidden_layers", 0)),
            layer_types=tuple(text.get("layer_types", [])),
            deepstack_indexes=tuple(deepstack),
            supports_interleaved_mrope=bool(rope.get("mrope_interleaved", False)),
        )

    @property
    def supports_delta_net(self) -> bool:
        return any(layer == "linear_attention" for layer in self.layer_types)


def read_checkpoint_spec(
    checkpoint: str | Path,
    *,
    model_id: str,
    revision: str | None,
) -> ModelSpec:
    path = Path(checkpoint) / "config.json"
    config = json.loads(path.read_text())
    return ModelSpec.from_config(model_id, config, revision=revision)
