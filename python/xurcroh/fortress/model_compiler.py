from __future__ import annotations

import json
import math
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from xurcroh.algebra.modulation import SignedPermutation
from xurcroh.errors import ConfigurationError
from xurcroh.formats import TensorReader
from xurcroh.fortress.combiner import FortressCombiner, FortressLayerMetadata
from xurcroh.fortress.compiler import CompiledOperation, compile_operation, reap_prune_experts
from xurcroh.fortress.placement import Operation


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


def _text_config(reader: TensorReader) -> dict[str, Any]:
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


def load_reap_saliency(path: str | Path) -> dict[int, dict[int, float]]:
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("REAP saliency must be a layer-to-expert object")
    saliency = {}
    for raw_layer, raw_experts in document.items():
        if not isinstance(raw_experts, dict):
            raise ValueError("REAP saliency must be a layer-to-expert object")
        scores = {}
        for raw_expert, raw_score in raw_experts.items():
            try:
                layer = int(raw_layer)
                expert = int(raw_expert)
                score = float(raw_score)
            except (TypeError, ValueError) as error:
                raise ValueError("REAP saliency indices and scores are invalid") from error
            if layer < 0 or expert < 0 or not math.isfinite(score) or score < 0:
                raise ValueError("REAP saliency indices and scores are invalid")
            scores[expert] = score
        saliency[layer] = scores
    return saliency


def _reap_expert_mapping(
    expert_indices: list[int],
    layer_index: int,
    keep_fraction: float | None,
    saliency: dict[int, dict[int, float]] | None,
) -> dict[int, int]:
    if keep_fraction is None:
        return {expert: expert for expert in expert_indices}
    if saliency is None or layer_index not in saliency:
        raise ValueError(f"REAP saliency is missing for layer {layer_index}")
    scores = saliency[layer_index]
    if set(scores) != set(expert_indices):
        raise ValueError(f"REAP saliency does not cover every expert in layer {layer_index}")
    pruned = reap_prune_experts(
        {str(expert): torch.empty(0) for expert in expert_indices},
        {str(expert): scores[expert] for expert in expert_indices},
        keep_fraction=keep_fraction,
    )
    selected = sorted(int(expert) for expert in pruned)
    return {expert: new_index for new_index, expert in enumerate(selected)}


def discover_fortress_operations(
    checkpoint: str | Path,
    *,
    reap_keep_fraction: float | None = None,
    reap_saliency: dict[int, dict[int, float]] | None = None,
) -> list[Operation]:
    from xurcroh.architectures import open_architecture

    adapter = open_architecture(checkpoint)
    reader = adapter.reader
    names = list(reader.tensor_names())
    shapes = reader.tensor_shapes(names)
    layer_family, layer_indices = _layer_family(names)
    discovered: list[Operation] = []

    def add(name: str) -> None:
        tensor_name = name if name.endswith(".weight") else name + ".weight"
        name = name.removesuffix(".weight")
        if tensor_name not in shapes or len(shapes[tensor_name]) != 2:
            return
        if ".self_attn." in name:
            kind = "attention"
        elif ".experts." in name:
            kind = "expert"
        else:
            kind = "feed_forward"
        paired_with: str | tuple[str, ...] | None = None
        if name.endswith(".q_proj"):
            paired_with = name.removesuffix("q_proj") + "k_proj"
        elif name.endswith(".k_proj"):
            paired_with = name.removesuffix("k_proj") + "q_proj"
        elif name.endswith((".gate_proj", ".up_proj", ".down_proj")):
            prefix = name.rsplit(".", 1)[0]
            paired_with = tuple(
                f"{prefix}.{projection}"
                for projection in ("gate_proj", "up_proj", "down_proj")
                if f"{prefix}.{projection}" != name
            )
        discovered.append(Operation(name, len(discovered), kind, paired_with))

    for index in layer_indices:
        prefix = f"{layer_family}.{index}"
        layer_names = [name for name in names if name.startswith(prefix + ".")]
        attention_prefix = f"{prefix}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"{attention_prefix}.{projection}.weight")
        mlp_prefix = f"{prefix}.mlp"
        expert_indices = sorted(
            {
                int(match.group(1))
                for name in layer_names
                if (match := re.search(r"mlp\.experts\.(\d+)\.", name))
            }
        )
        if expert_indices:
            expert_mapping = _reap_expert_mapping(
                expert_indices, index, reap_keep_fraction, reap_saliency
            )
            for original_expert, expert in expert_mapping.items():
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    source = f"{mlp_prefix}.experts.{original_expert}.{projection}.weight"
                    add(source.replace(f".experts.{original_expert}.", f".experts.{expert}."))
        else:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                add(f"{mlp_prefix}.{projection}.weight")
        selected = {operation.identifier + ".weight" for operation in discovered}
        for name in layer_names:
            operation_name = name.removesuffix(".weight")
            if (
                name.endswith(".weight")
                and len(shapes[name]) == 2
                and name not in selected
                and ".per_layer_" not in name
                and ".router." not in name
                and ".experts." not in name
                and name != f"{mlp_prefix}.gate.weight"
            ):
                add(operation_name)
    return discovered


def _derived_seed(seed: int, operation: str, purpose: str) -> bytes:
    if not 0 <= seed < 2**256:
        raise ValueError("Fortress seed must fit in 256 bits")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"xurcroh-fortress-v1",
        info=f"{purpose}:{operation}".encode(),
    ).derive(seed.to_bytes(32, "big"))


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
    retain_quantized: bool = True,
    operation_sink: Callable[[str, CompiledOperation], None] | None = None,
) -> None:
    if name in operations:
        raise ConfigurationError(f"duplicate Fortress operation: {name}")
    input_basis = None
    output_basis = None
    if inner_bases:
        input_seed = (
            int.from_bytes(_derived_seed(seed, name, "input-basis")) % 2**63
            if seed is not None
            else secrets.randbits(63)
        )
        output_seed = (
            int.from_bytes(_derived_seed(seed, name, "output-basis")) % 2**63
            if seed is not None
            else secrets.randbits(63)
        )
        input_basis = SignedPermutation.generate(weight.shape[1], seed=input_seed)
        output_basis = SignedPermutation.generate(weight.shape[0], seed=output_seed)
    share_seed = _derived_seed(seed, name, "weight-share") if seed is not None else None
    operation = compile_operation(
        weight.to(torch.float64),
        threshold=threshold,
        slice_count=slice_count,
        share_count=share_count,
        seed=share_seed,
        input_basis=input_basis,
        output_basis=output_basis,
        materialize_shares=materialize_shares,
    )
    if operation_sink is not None:
        operation_sink(name, operation)
        operation = replace(
            operation,
            shares={
                index: replace(share, matrix=None) for index, share in operation.shares.items()
            },
            storage={},
        )
    quantized = replace(
        operation.quantized,
        values=operation.quantized.require_values().to(torch.int8),
    )
    if not retain_quantized:
        quantized = replace(quantized, values=None, shape=quantized.dimensions)
    operations[name] = replace(operation, quantized=quantized)


def _boundary_tensor(
    reader: TensorReader, names: list[str], candidates: tuple[str, ...]
) -> torch.Tensor:
    for candidate in candidates:
        if candidate in names:
            return reader.read_tensors([candidate])[candidate].to(torch.float64)
    raise ConfigurationError(f"cannot identify checkpoint boundary tensor: {candidates}")


def _optional_media_projection(
    reader: TensorReader, names: list[str], candidates: tuple[str, ...]
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
    retain_quantized: bool = True,
    model_dtype: torch.dtype | None = None,
    load_model: bool = True,
    load_full_multimodal: bool = False,
    reap_keep_fraction: float | None = None,
    reap_saliency: dict[int, dict[int, float]] | None = None,
    operation_sink: Callable[[str, CompiledOperation], None] | None = None,
) -> FortressCheckpoint:
    if operation_sink is not None and not materialize_shares:
        raise ValueError("role packaging requires materialized weight shares")
    from xurcroh.architectures import open_architecture

    adapter = open_architecture(checkpoint)
    reader = adapter.reader
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
        expert_mapping = _reap_expert_mapping(
            expert_indices, index, reap_keep_fraction, reap_saliency
        )
        if expert_indices:
            layer_names = [
                name
                for name in layer_names
                if (match := re.search(r"mlp\.experts\.(\d+)\.", name)) is None
                or int(match.group(1)) in expert_mapping
            ]
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
                retain_quantized=retain_quantized,
                operation_sink=operation_sink,
            )
        mlp_prefix = f"{prefix}.mlp"
        if expert_indices:
            for original_expert, expert in expert_mapping.items():
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    _compile_matrix(
                        f"{mlp_prefix}.experts.{expert}.{projection}",
                        tensors[f"{mlp_prefix}.experts.{original_expert}.{projection}.weight"],
                        operations,
                        threshold=threshold,
                        slice_count=slice_count,
                        inner_bases=inner_bases,
                        seed=seed,
                        share_count=share_count,
                        materialize_shares=materialize_shares,
                        retain_quantized=retain_quantized,
                        operation_sink=operation_sink,
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
                    retain_quantized=retain_quantized,
                    operation_sink=operation_sink,
                )
        for name, tensor in tensors.items():
            operation_name = name.removesuffix(".weight")
            if (
                name.endswith(".weight")
                and tensor.ndim == 2
                and operation_name not in operations
                and ".per_layer_" not in name
                and ".router." not in name
                and ".experts." not in name
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
                    retain_quantized=retain_quantized,
                    operation_sink=operation_sink,
                )
        router = (
            tensors[f"{mlp_prefix}.gate.weight"][list(expert_mapping)]
            .to(torch.float64)
            .contiguous()
            if expert_indices
            else None
        )
        top_k = min(
            int(config.get("num_experts_per_tok", config.get("moe_top_k", config.get("top_k", 1)))),
            len(expert_mapping) if expert_indices else 1,
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
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model = None
    if load_model and reader.config.get("model_type") in CONFIG_MAPPING:
        if load_full_multimodal:
            if reader.config.get("model_type") not in {"gemma4", "qwen3_5"}:
                raise ConfigurationError(
                    "full Fortress VLM loading supports Gemma 4 and Qwen 3.5 checkpoints only"
                )
            from xurcroh.fortress.vlm_loader import load_multimodal_decoder

            model, model_operations = load_multimodal_decoder(
                str(checkpoint), operations, boundary_dtype
            )
        elif reader.source_format != "safetensors" or reader.config.get("model_type") in {
            "gemma4",
            "qwen3_5",
        }:
            from xurcroh.fortress.text_loader import load_text_decoder

            model, model_operations = load_text_decoder(reader, operations, boundary_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=boundary_dtype,
                local_files_only=True,
                trust_remote_code=False,
            )
            model_operations = operations
        from xurcroh.fortress.module import replace_linear_layers
        from xurcroh.fortress.text_loader import CLIENT_ONLY

        if reap_keep_fraction is not None:
            for index, layer in enumerate(layers):
                if layer.router is None:
                    continue
                mlp = model.model.layers[index].mlp
                mlp.gate.weight = torch.nn.Parameter(
                    layer.router.to(mlp.gate.weight.dtype), requires_grad=False
                )
                mlp.gate.num_experts = layer.router.shape[0]
                mlp.gate.top_k = layer.top_k
                mlp.num_experts = layer.router.shape[0]
        if not load_full_multimodal and reader.config.get("model_type") not in {
            "gemma4",
            "qwen3_5",
        }:
            replace_linear_layers(model, model_operations, client_only=CLIENT_ONLY)
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
