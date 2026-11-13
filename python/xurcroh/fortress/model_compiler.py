from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from xurcroh.algebra.modulation import SignedPermutation
from xurcroh.errors import ConfigurationError
from xurcroh.fortress.combiner import FortressCombiner, FortressLayerMetadata
from xurcroh.fortress.compiler import CompiledOperation, compile_operation
from xurcroh.shards.compiler import CheckpointReader


@dataclass(frozen=True)
class FortressCheckpoint:
    layers: tuple[FortressLayerMetadata, ...]
    operations: dict[str, CompiledOperation]
    embedding: torch.Tensor
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    query_heads: int
    kv_heads: int
    rope_theta: float
    activation_bits: int
    media_projections: dict[str, torch.Tensor] = field(default_factory=dict)
    model: Any = field(default=None, repr=False)

    def combiner(self) -> FortressCombiner:
        return FortressCombiner(
            layers=list(self.layers),
            operations=self.operations,
            query_heads=self.query_heads,
            kv_heads=self.kv_heads,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            rope_theta=self.rope_theta,
            activation_bits=self.activation_bits,
        )


def _text_config(reader: CheckpointReader) -> dict[str, Any]:
    return {**reader.config, **reader.config.get("text_config", {})}


def _layer_family(names: list[str]) -> tuple[str, list[int]]:
    families: dict[str, set[int]] = {}
    for name in names:
        match = re.search(r"((?:^|.*\.)layers)\.(\d+)\.", name)
        if match is not None:
            families.setdefault(match.group(1), set()).add(int(match.group(2)))
    if not families:
        raise ConfigurationError("checkpoint contains no transformer layers")
    family = max(families, key=lambda name: len(families[name]))
    return family, sorted(families[family])


def _compile_matrix(
    name: str,
    weight: torch.Tensor,
    operations: dict[str, CompiledOperation],
    *,
    threshold: int,
    slice_count: int,
    inner_bases: bool,
    seed: int | None,
    share_count: int = 2,
    materialize_shares: bool = True,
) -> None:
    if name in operations:
        raise ConfigurationError(f"duplicate Fortress operation: {name}")
    input_basis = None
    output_basis = None
    if inner_bases:
        operation_index = len(operations)
        input_seed = seed + operation_index * 2 if seed is not None else secrets.randbits(63)
        output_seed = seed + operation_index * 2 + 1 if seed is not None else secrets.randbits(63)
        input_basis = SignedPermutation.generate(weight.shape[1], seed=input_seed)
        output_basis = SignedPermutation.generate(weight.shape[0], seed=output_seed)
    share_seed = hashlib.sha256(f"{seed}:{name}".encode()).digest() if seed is not None else None
    operation = compile_operation(
        weight.to(torch.float64),
        threshold=threshold,
        slice_count=slice_count,
        share_count=share_count,
        seed=share_seed,
        challenge_seed=seed + len(operations) if seed is not None else None,
        input_basis=input_basis,
        output_basis=output_basis,
        materialize_shares=materialize_shares,
    )
    operations[name] = replace(
        operation,
        quantized=replace(operation.quantized, values=operation.quantized.values.to(torch.int8)),
    )


def _boundary_tensor(
    reader: CheckpointReader, names: list[str], candidates: tuple[str, ...]
) -> torch.Tensor:
    for candidate in candidates:
        if candidate in names:
            return reader.read_tensors([candidate])[candidate].to(torch.float64)
    raise ConfigurationError(f"cannot identify checkpoint boundary tensor: {candidates}")


def _optional_media_projection(
    reader: CheckpointReader, names: list[str], candidates: tuple[str, ...]
) -> torch.Tensor | None:
    for candidate in candidates:
        if candidate in names:
            return reader.read_tensors([candidate])[candidate].to(torch.float64)
    return None


def compile_fortress_checkpoint(
    checkpoint: str | Path,
    *,
    threshold: int = 2,
    slice_count: int = 3,
    inner_bases: bool = True,
    activation_bits: int = 12,
    seed: int | None = None,
    share_count: int = 2,
    materialize_shares: bool = True,
    model_dtype: torch.dtype | None = None,
    load_model: bool = True,
) -> FortressCheckpoint:
    reader = CheckpointReader(Path(checkpoint))
    config = _text_config(reader)
    names = list(reader.tensor_names())
    layer_family, layer_indices = _layer_family(names)
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", query_heads))
    head_dimension = int(config.get("head_dim", config["hidden_size"] // query_heads))
    layers: list[FortressLayerMetadata] = []
    operations: dict[str, CompiledOperation] = {}
    for index in layer_indices:
        prefix = f"{layer_family}.{index}"
        layer_names = [name for name in names if name.startswith(prefix + ".")]
        expert_indices = sorted(
            {
                int(match.group(1))
                for name in layer_names
                if (match := re.search(r"mlp\.experts\.(\d+)\.", name))
            }
        )
        if expert_indices != list(range(len(expert_indices))):
            raise ConfigurationError("expert indices must be contiguous and zero-based")
        tensors = reader.read_tensors(layer_names)
        attention_prefix = f"{prefix}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"{attention_prefix}.{projection}"
            weight = tensors.get(f"{name}.weight")
            if weight is None:
                continue
            _compile_matrix(
                name,
                weight,
                operations,
                threshold=threshold,
                slice_count=slice_count,
                inner_bases=inner_bases,
                seed=seed,
                share_count=share_count,
                materialize_shares=materialize_shares,
            )
        mlp_prefix = f"{prefix}.mlp"
        if expert_indices:
            for expert in expert_indices:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    _compile_matrix(
                        f"{mlp_prefix}.experts.{expert}.{projection}",
                        tensors[f"{mlp_prefix}.experts.{expert}.{projection}.weight"],
                        operations,
                        threshold=threshold,
                        slice_count=slice_count,
                        inner_bases=inner_bases,
                        seed=seed,
                        share_count=share_count,
                        materialize_shares=materialize_shares,
                    )
        else:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                _compile_matrix(
                    f"{mlp_prefix}.{projection}",
                    tensors[f"{mlp_prefix}.{projection}.weight"],
                    operations,
                    threshold=threshold,
                    slice_count=slice_count,
                    inner_bases=inner_bases,
                    seed=seed,
                    share_count=share_count,
                    materialize_shares=materialize_shares,
                )
        for name, tensor in tensors.items():
            operation_name = name.removesuffix(".weight")
            if (
                name.endswith(".weight")
                and tensor.ndim == 2
                and operation_name not in operations
                and ".per_layer_" not in name
                and ".router." not in name
                and name != f"{mlp_prefix}.gate.weight"
            ):
                _compile_matrix(
                    operation_name,
                    tensor,
                    operations,
                    threshold=threshold,
                    slice_count=slice_count,
                    inner_bases=inner_bases,
                    seed=seed,
                    share_count=share_count,
                    materialize_shares=materialize_shares,
                )
        router = tensors[f"{mlp_prefix}.gate.weight"].to(torch.float64) if expert_indices else None
        top_k = int(
            config.get("num_experts_per_tok", config.get("moe_top_k", config.get("top_k", 1)))
        )
        layers.append(
            FortressLayerMetadata(
                head_dimension=head_dimension,
                q_norm=tensors.get(f"{attention_prefix}.q_norm.weight", None),
                k_norm=tensors.get(f"{attention_prefix}.k_norm.weight", None),
                input_gain=tensors[f"{prefix}.input_layernorm.weight"].to(torch.float64),
                post_attention_gain=tensors[f"{prefix}.post_attention_layernorm.weight"].to(
                    torch.float64
                ),
                router=router,
                top_k=top_k,
            )
        )
        for tensor in tensors.values():
            tensor.zero_()
    embedding = _boundary_tensor(
        reader,
        names,
        (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
        ),
    )
    final_norm = _boundary_tensor(
        reader,
        names,
        ("model.norm.weight", "model.language_model.norm.weight", "transformer.ln_f.weight"),
    )
    head_name = next(
        (name for name in ("lm_head.weight", "model.lm_head.weight") if name in names),
        None,
    )
    lm_head = embedding if head_name is None else reader.read_tensors([head_name])[head_name]
    boundary_dtype = model_dtype or torch.float64
    embedding = embedding.to(boundary_dtype)
    lm_head = lm_head.to(boundary_dtype)
    media_projections: dict[str, torch.Tensor] = {}
    vision = _optional_media_projection(
        reader,
        names,
        (
            "vision_projection.weight",
            "model.vision_projection.weight",
            "multi_modal_projector.weight",
        ),
    )
    audio = _optional_media_projection(
        reader,
        names,
        ("audio_projection.weight", "model.audio_projection.weight", "model.embed_audio.weight"),
    )
    if vision is not None:
        media_projections["vision"] = vision
    if audio is not None:
        media_projections["audio"] = audio
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model = None
    if load_model and reader.config.get("model_type") in CONFIG_MAPPING:
        loader = (
            AutoModelForMultimodalLM
            if reader.config.get("model_type") in {"gemma4", "qwen3_5"}
            else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            checkpoint,
            dtype=boundary_dtype,
            local_files_only=True,
            trust_remote_code=False,
        )
        from xurcroh.fortress.module import replace_linear_layers

        replace_linear_layers(
            model,
            operations,
            client_only=(
                "lm_head",
                "model.per_layer_model_projection",
                "model.language_model.per_layer_model_projection",
                "per_layer_input_gate",
                "per_layer_projection",
                "vision_projection",
                "model.vision_projection",
                "multi_modal_projector",
                "audio_projection",
                "model.audio_projection",
                "model.embed_audio",
            ),
        )
        model.get_input_embeddings().weight = torch.nn.Parameter(embedding, requires_grad=False)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight = torch.nn.Parameter(lm_head, requires_grad=False)
        model.eval()
    return FortressCheckpoint(
        layers=tuple(layers),
        operations=operations,
        embedding=embedding,
        final_norm=final_norm,
        lm_head=lm_head,
        query_heads=query_heads,
        kv_heads=kv_heads,
        rope_theta=float(config.get("rope_theta", 10_000)),
        activation_bits=activation_bits,
        media_projections=media_projections,
        model=model,
    )
