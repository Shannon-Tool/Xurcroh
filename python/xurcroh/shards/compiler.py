from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from xurcroh.algebra.attention import (
    AttentionWeights,
    conjugate_attention,
    rope_commuting_pair,
    sample_orthogonal,
)
from xurcroh.algebra.feedforward import FeedForwardWeights, conjugate_feedforward
from xurcroh.algebra.rmsnorm import absorb_rms_gain
from xurcroh.errors import ConfigurationError, ModelLoadError
from xurcroh.shards.format import write_shard
from xurcroh.shards.memory import release_host_memory
from xurcroh.shards.recipients import StageRecipient, manifest_digest, seal_stage_key

LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class CompilationConfig:
    checkpoint: Path
    output_dir: Path
    stage_count: int
    boundary_key: bytes
    recipients: tuple[StageRecipient, ...]
    seed: int | None = None
    compute_dtype: torch.dtype = torch.float64
    shard_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.stage_count < 1:
            raise ValueError("stage_count must be positive")
        if len(self.boundary_key) != 32:
            raise ValueError("boundary_key must contain 32 bytes")
        if len(self.recipients) != self.stage_count:
            raise ValueError("each stage requires exactly one recipient")
        if tuple(recipient.stage_index for recipient in self.recipients) != tuple(
            range(self.stage_count)
        ):
            raise ValueError("recipient stage indexes must be sequential")
        if self.shard_dtype not in {
            None,
            torch.float64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise ValueError("shard_dtype must be FP64, FP32, FP16, or BF16")

    @property
    def storage_dtype(self) -> torch.dtype:
        return self.shard_dtype or self.compute_dtype


@dataclass(frozen=True)
class CompilationResult:
    manifest_path: Path
    boundary_shard: Path
    stage_shards: tuple[Path, ...]
    key_envelopes: tuple[Path, ...]


class CheckpointReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        if not root.is_dir():
            raise ModelLoadError(f"checkpoint directory does not exist: {root}")
        config_path = root / "config.json"
        try:
            self.config_bytes = config_path.read_bytes()
            self.config = json.loads(self.config_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ModelLoadError(f"cannot read checkpoint config: {config_path}") from error
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self.weight_map = dict(index["weight_map"])
            self._index_bytes = index_path.read_bytes()
        elif (root / "model.safetensors").is_file():
            self.weight_map = {}
            with safe_open(root / "model.safetensors", framework="pt") as handle:
                tensor_names = handle.keys()
                for name in tensor_names:
                    self.weight_map[name] = "model.safetensors"
            self._index_bytes = b""
        else:
            raise ModelLoadError("strict compiler requires safetensors checkpoint files")
        self._index_bytes = self.config_bytes + self._index_bytes

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self._index_bytes).hexdigest()

    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        grouped: dict[str, list[str]] = {}
        for name in names:
            if name not in self.weight_map:
                raise ModelLoadError(f"checkpoint tensor is missing: {name}")
            grouped.setdefault(self.weight_map[name], []).append(name)
        tensors: dict[str, torch.Tensor] = {}
        for filename, requested in grouped.items():
            with safe_open(self.root / filename, framework="pt") as handle:
                for name in requested:
                    tensors[name] = handle.get_tensor(name)
        return tensors


def partition_layers(layer_indices: list[int], stage_count: int) -> list[list[int]]:
    if not layer_indices or stage_count > len(layer_indices):
        raise ConfigurationError("stage_count cannot exceed transformer layer count")
    expected = list(range(len(layer_indices)))
    if sorted(layer_indices) != expected:
        raise ConfigurationError("transformer layer indices must be contiguous and zero-based")
    base, remainder = divmod(len(layer_indices), stage_count)
    partitions = []
    offset = 0
    for stage in range(stage_count):
        size = base + (1 if stage < remainder else 0)
        partitions.append(layer_indices[offset : offset + size])
        offset += size
    return partitions


class XurcrohWeightCompiler:
    def __init__(self, config: CompilationConfig) -> None:
        self.config = config
        self.reader = CheckpointReader(config.checkpoint)

    def _text_config(self) -> dict[str, Any]:
        nested = self.reader.config.get("text_config", {})
        return {**self.reader.config, **nested}

    def _compile_dense_layer(
        self,
        tensors: dict[str, torch.Tensor],
        prefix: str,
        basis: torch.Tensor,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        dtype = self.config.compute_dtype
        text_config = self._text_config()
        query_heads = int(text_config["num_attention_heads"])
        kv_heads = int(text_config["num_key_value_heads"])
        hidden_size = int(text_config["hidden_size"])
        head_dimension = int(text_config.get("head_dim", hidden_size // query_heads))
        layer = {name: tensor.to(dtype) for name, tensor in tensors.items()}
        output = dict(layer)
        attention_prefix = f"{prefix}.self_attn"
        required = [
            f"{attention_prefix}.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        if not all(name in layer for name in required):
            raise ConfigurationError(f"unsupported attention layout in {prefix}")
        known_attention = set(required)
        known_attention.update(
            f"{attention_prefix}.{name}.bias" for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        q_norm_name = f"{attention_prefix}.q_norm.weight"
        k_norm_name = f"{attention_prefix}.k_norm.weight"
        has_qk_norm = q_norm_name in layer or k_norm_name in layer
        if has_qk_norm and not (q_norm_name in layer and k_norm_name in layer):
            raise ConfigurationError("Qwen-style attention requires both q_norm and k_norm")
        known_attention.update((q_norm_name, k_norm_name))
        for name in layer:
            if name.startswith(attention_prefix) and name not in known_attention:
                raise ConfigurationError(f"unsupported attention tensor: {name}")

        gain_name = f"{prefix}.input_layernorm.weight"
        ffn_gain_name = f"{prefix}.post_attention_layernorm.weight"
        gain = layer[gain_name]
        attention = AttentionWeights(
            q=absorb_rms_gain(layer[required[0]], gain),
            k=absorb_rms_gain(layer[required[1]], gain),
            v=absorb_rms_gain(layer[required[2]], gain),
            o=layer[required[3]],
            q_norm=layer.get(q_norm_name),
            k_norm=layer.get(k_norm_name),
        )
        q_transform, k_transform = rope_commuting_pair(
            query_heads,
            kv_heads,
            head_dimension,
            seed=seed,
            minimum_scale=1.0 if has_qk_norm else 0.6,
            maximum_scale=1.0 if has_qk_norm else 1.6,
            interleaved=text_config.get("rope_style") == "interleaved",
            preserve_rms_norm=has_qk_norm,
            shared_heads=has_qk_norm,
        )
        q_transform = q_transform.to(dtype)
        k_transform = k_transform.to(dtype)
        generator = torch.Generator().manual_seed(seed + 1)
        value_blocks = []
        for _ in range(kv_heads):
            left = sample_orthogonal(
                head_dimension,
                seed=int(torch.randint(1, 1_000_000, (1,), generator=generator).item()),
                dtype=dtype,
            )
            right = sample_orthogonal(
                head_dimension,
                seed=int(torch.randint(1, 1_000_000, (1,), generator=generator).item()),
                dtype=dtype,
            )
            scales = torch.empty(head_dimension, dtype=dtype)
            scales.uniform_(0.6, 1.6, generator=generator)
            value_blocks.append(left @ torch.diag(scales) @ right)
        value_transform = torch.block_diag(*value_blocks)  # type: ignore[no-untyped-call]
        output_transform = torch.block_diag(  # type: ignore[no-untyped-call]
            *[value_blocks[index // (query_heads // kv_heads)] for index in range(query_heads)]
        )
        transformed = conjugate_attention(
            attention,
            basis,
            basis,
            q_transform,
            k_transform,
            value_transform,
            torch.linalg.inv(output_transform),
        )
        for name, value in zip(
            required, (transformed.q, transformed.k, transformed.v, transformed.o), strict=True
        ):
            output[name] = value
        if transformed.q_norm is not None:
            output[q_norm_name] = transformed.q_norm
        if transformed.k_norm is not None:
            output[k_norm_name] = transformed.k_norm
        for projection, transform in (
            ("q_proj", q_transform),
            ("k_proj", k_transform),
            ("v_proj", value_transform),
        ):
            bias_name = f"{attention_prefix}.{projection}.bias"
            if bias_name in layer:
                output[bias_name] = transform @ layer[bias_name]
        output[gain_name] = torch.ones_like(layer[gain_name])

        mlp_prefix = f"{prefix}.mlp"
        ffn_names = [
            f"{mlp_prefix}.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if all(name in layer for name in ffn_names):
            ffn = FeedForwardWeights(
                gate=absorb_rms_gain(layer[ffn_names[0]], layer[ffn_gain_name]),
                up=absorb_rms_gain(layer[ffn_names[1]], layer[ffn_gain_name]),
                down=layer[ffn_names[2]],
            )
            intermediate = ffn.gate.shape[0]
            permutation = torch.randperm(intermediate, generator=generator)
            scales = torch.empty(intermediate, dtype=dtype)
            scales.uniform_(0.5, 2.0, generator=generator)
            transformed_ffn = conjugate_feedforward(ffn, basis, basis, permutation, scales)
            output[ffn_names[0]] = transformed_ffn.gate
            output[ffn_names[1]] = transformed_ffn.up
            output[ffn_names[2]] = transformed_ffn.down
        elif any(name in layer for name in ffn_names):
            raise ConfigurationError(f"incomplete dense FFN projections in {prefix}")
        output[ffn_gain_name] = torch.ones_like(layer[ffn_gain_name])
        return output

    def _compile_moe_layer(
        self,
        tensors: dict[str, torch.Tensor],
        prefix: str,
        basis: torch.Tensor,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        mlp_prefix = f"{prefix}.mlp"
        expert_indices = sorted(
            {
                int(match.group(1))
                for name in tensors
                if (match := re.search(r"mlp\.experts\.(\d+)\.", name))
            }
        )
        if expert_indices != list(range(len(expert_indices))):
            raise ConfigurationError("expert indices must be contiguous and zero-based")
        output = self._compile_dense_layer(tensors, prefix, basis, seed)
        for name in list(output):
            if name.startswith(f"{mlp_prefix}.experts.") or name == f"{mlp_prefix}.gate.weight":
                del output[name]
        generator = torch.Generator().manual_seed(seed)
        expert_permutation = torch.randperm(len(expert_indices), generator=generator)
        router_name = f"{mlp_prefix}.gate.weight"
        gain = tensors[f"{prefix}.post_attention_layernorm.weight"].to(self.config.compute_dtype)
        router = absorb_rms_gain(tensors[router_name].to(self.config.compute_dtype), gain)
        output[router_name] = router[expert_permutation] @ basis.T
        output[f"{mlp_prefix}.expert_permutation"] = expert_permutation
        for new_index, original_position in enumerate(expert_permutation.tolist()):
            original_expert = expert_indices[original_position]
            original_prefix = f"{mlp_prefix}.experts.{original_expert}"
            target_prefix = f"{mlp_prefix}.experts.{new_index}"
            expert = FeedForwardWeights(
                gate=absorb_rms_gain(
                    tensors[f"{original_prefix}.gate_proj.weight"].to(self.config.compute_dtype),
                    gain,
                ),
                up=absorb_rms_gain(
                    tensors[f"{original_prefix}.up_proj.weight"].to(self.config.compute_dtype), gain
                ),
                down=tensors[f"{original_prefix}.down_proj.weight"].to(self.config.compute_dtype),
            )
            expert_generator = torch.Generator().manual_seed(seed + new_index + 17)
            intermediate = expert.gate.shape[0]
            permutation = torch.randperm(intermediate, generator=expert_generator)
            scales = torch.empty(intermediate, dtype=self.config.compute_dtype)
            scales.uniform_(0.5, 2.0, generator=expert_generator)
            transformed = conjugate_feedforward(expert, basis, basis, permutation, scales)
            output[f"{target_prefix}.gate_proj.weight"] = transformed.gate
            output[f"{target_prefix}.up_proj.weight"] = transformed.up
            output[f"{target_prefix}.down_proj.weight"] = transformed.down
        return output

    def _compile_layer(
        self,
        tensors: dict[str, torch.Tensor],
        prefix: str,
        basis: torch.Tensor,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        if any(name.startswith(f"{prefix}.mlp.experts.") for name in tensors):
            return self._compile_moe_layer(tensors, prefix, basis, seed)
        return self._compile_dense_layer(tensors, prefix, basis, seed)

    def compile(self) -> CompilationResult:
        seed = self.config.seed if self.config.seed is not None else secrets.randbits(63)
        text_config = self._text_config()
        hidden_size = int(text_config["hidden_size"])
        query_heads = int(text_config["num_attention_heads"])
        kv_heads = int(text_config["num_key_value_heads"])
        names = list(self.reader.tensor_names())
        layer_indices = sorted(
            {int(match.group(1)) for name in names if (match := LAYER_PATTERN.search(name))}
        )
        partitions = partition_layers(layer_indices, self.config.stage_count)
        bases = [
            sample_orthogonal(
                hidden_size, seed=seed + 1009 * index, dtype=self.config.compute_dtype
            )
            for index in range(self.config.stage_count)
        ]
        embedding_name = next(
            (
                name
                for name in (
                    "model.embed_tokens.weight",
                    "embed_tokens.weight",
                    "transformer.wte.weight",
                )
                if name in names
            ),
            None,
        )
        head_name = next(
            (name for name in ("lm_head.weight", "model.lm_head.weight") if name in names), None
        )
        if embedding_name is None:
            raise ConfigurationError("cannot identify token embedding boundary")
        final_norm_name = next(
            (name for name in ("model.norm.weight", "transformer.ln_f.weight") if name in names),
            None,
        )
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stage_shards: list[Path] = []
        key_envelopes: list[Path] = []
        stage_keys: list[bytes] = []
        manifest_stages = []
        for stage_index, indices in enumerate(partitions):
            recipient = self.config.recipients[stage_index]
            basis = bases[stage_index]
            compiled: dict[str, torch.Tensor] = {}
            for index in indices:
                prefix = f"model.layers.{index}"
                layer_names = [
                    name
                    for name in names
                    if name.startswith(prefix + ".") and name != final_norm_name
                ]
                layer_tensors = self.reader.read_tensors(layer_names)
                transformed = self._compile_layer(
                    layer_tensors,
                    prefix,
                    basis,
                    seed + index * 97 + stage_index,
                )
                converted = {
                    name: tensor.to(self.config.storage_dtype)
                    if tensor.is_floating_point()
                    else tensor
                    for name, tensor in transformed.items()
                }
                if self.config.storage_dtype != self.config.compute_dtype:
                    for tensor in transformed.values():
                        tensor.zero_()
                transformed.clear()
                compiled.update(converted)
                for tensor in layer_tensors.values():
                    tensor.zero_()
                layer_tensors.clear()
                release_host_memory()
            metadata = {
                "role": "worker-stage",
                "stage_index": stage_index,
                "stage_count": self.config.stage_count,
                "layer_indices": indices,
                "model_type": self.reader.config.get("model_type", "unknown"),
                "storage_dtype": {
                    torch.float64: "f64",
                    torch.float32: "f32",
                    torch.float16: "f16",
                    torch.bfloat16: "bf16",
                }[self.config.storage_dtype],
                "interleaved_rope": text_config.get("rope_style") == "interleaved",
                "rope_theta": float(
                    text_config.get(
                        "rope_theta",
                        text_config.get("rope_parameters", {}).get("rope_theta", 10_000),
                    )
                ),
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "moe_top_k": int(text_config.get("num_experts_per_tok", 1)),
                "transformation": "dense-contragredient-v1",
                "checkpoint_fingerprint": self.reader.fingerprint,
            }
            compiled = {
                name: tensor.to(self.config.storage_dtype) if tensor.is_floating_point() else tensor
                for name, tensor in compiled.items()
            }
            shard_path = output_dir / f"stage-{stage_index}.xurcroh"
            stage_key = os.urandom(32)
            shard_metadata = write_shard(shard_path, compiled, stage_key, metadata)
            key_path = output_dir / f"stage-{stage_index}.key.json"
            stage_keys.append(stage_key)
            stage_shards.append(shard_path)
            key_envelopes.append(key_path)
            manifest_stages.append(
                {
                    **shard_metadata,
                    "file": shard_path.name,
                    "key_envelope": key_path.name,
                    "key_id": recipient.key_id,
                    "recipient_public": recipient.public_key.hex(),
                }
            )
            for tensor in compiled.values():
                tensor.zero_()
            compiled.clear()
            release_host_memory()

        boundary_names = [embedding_name]
        if head_name is not None:
            boundary_names.append(head_name)
        if final_norm_name is not None:
            boundary_names.append(final_norm_name)
        boundary = self.reader.read_tensors(boundary_names)
        if head_name is None:
            shared_embedding = boundary[embedding_name].to(self.config.storage_dtype)
            boundary_tensors = {
                "embedding.weight": shared_embedding,
                "lm_head.weight": shared_embedding,
            }
        else:
            embedding = boundary[embedding_name].to(self.config.compute_dtype)
            lm_head = boundary[head_name].to(self.config.compute_dtype)
            if final_norm_name is not None:
                lm_head = absorb_rms_gain(
                    lm_head, boundary[final_norm_name].to(self.config.compute_dtype)
                )
            boundary_tensors = {
                "embedding.weight": embedding.to(self.config.storage_dtype),
                "lm_head.weight": lm_head.to(self.config.storage_dtype),
            }
            del embedding, lm_head
        boundary_tensors.update(
            {
                "basis.input": bases[0].to(self.config.compute_dtype),
                "basis.output": bases[-1].to(self.config.compute_dtype),
                "basis.stages": torch.stack(bases).to(self.config.compute_dtype),
                "final_norm.epsilon": torch.tensor(
                    float(text_config.get("rms_norm_eps", 1e-6)),
                    dtype=self.config.storage_dtype,
                ),
            }
        )
        if final_norm_name is not None:
            boundary_tensors["final_norm.weight"] = torch.ones_like(
                boundary[final_norm_name], dtype=self.config.storage_dtype
            )
        release_host_memory()
        boundary_path = output_dir / "client-boundary.xurcroh"
        boundary_metadata = write_shard(
            boundary_path,
            boundary_tensors,
            self.config.boundary_key,
            {
                "role": "client-boundary",
                "tied_embeddings": head_name is None,
                "checkpoint_fingerprint": self.reader.fingerprint,
            },
        )
        for tensor in boundary_tensors.values():
            tensor.zero_()
        boundary_tensors.clear()
        for tensor in boundary.values():
            tensor.zero_()
        boundary.clear()
        for basis in bases:
            basis.zero_()
        manifest = {
            "format": "xurcroh-manifest",
            "version": 2,
            "model_type": self.reader.config.get("model_type", "unknown"),
            "checkpoint_fingerprint": self.reader.fingerprint,
            "stage_count": self.config.stage_count,
            "layer_count": len(layer_indices),
            "stages": manifest_stages,
            "boundary": {
                **boundary_metadata,
                "file": boundary_path.name,
                "tied_embeddings": head_name is None,
            },
        }
        digest = manifest_digest(manifest)
        manifest["digest"] = digest
        for stage_index, (key_path, stage_key) in enumerate(
            zip(key_envelopes, stage_keys, strict=True)
        ):
            envelope = seal_stage_key(
                stage_key,
                self.config.recipients[stage_index],
                manifest_digest=digest,
                checkpoint_fingerprint=self.reader.fingerprint,
            )
            key_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return CompilationResult(
            manifest_path=manifest_path,
            boundary_shard=boundary_path,
            stage_shards=tuple(stage_shards),
            key_envelopes=tuple(key_envelopes),
        )
