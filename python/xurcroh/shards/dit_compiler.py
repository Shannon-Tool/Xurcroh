from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from xurcroh.algebra.attention import AttentionWeights, conjugate_attention, sample_orthogonal
from xurcroh.algebra.feedforward import FeedForwardWeights, conjugate_feedforward
from xurcroh.algebra.modulation import SignedPermutation
from xurcroh.models.dit import DiTBlockWeights, DiTStageWeights, ModulationWeights
from xurcroh.shards.format import write_shard
from xurcroh.shards.recipients import (
    StageRecipient,
    manifest_digest,
    seal_stage_key,
)


@dataclass(frozen=True)
class DitCompilationResult:
    manifest_path: Path
    stage_shard: Path
    boundary_shard: Path
    key_envelope: Path


def _signed_matrix(transform: SignedPermutation) -> torch.Tensor:
    matrix = torch.zeros(
        transform.permutation.numel(),
        transform.permutation.numel(),
        dtype=torch.float64,
    )
    matrix[torch.arange(transform.permutation.numel()), transform.permutation] = transform.signs.to(
        matrix.dtype
    )
    return matrix


def _attention(tensors: dict[str, torch.Tensor], prefix: str) -> AttentionWeights:
    return AttentionWeights(
        q=tensors[f"{prefix}.q"],
        k=tensors[f"{prefix}.k"],
        v=tensors[f"{prefix}.v"],
        o=tensors[f"{prefix}.o"],
    )


def _ffn(tensors: dict[str, torch.Tensor], prefix: str) -> FeedForwardWeights:
    return FeedForwardWeights(
        gate=tensors[f"{prefix}.gate"],
        up=tensors[f"{prefix}.up"],
        down=tensors[f"{prefix}.down"],
    )


def _modulation(tensors: dict[str, torch.Tensor], prefix: str) -> ModulationWeights:
    return ModulationWeights(
        gamma=tensors[f"{prefix}.gamma"],
        beta=tensors[f"{prefix}.beta"],
        alpha=tensors[f"{prefix}.alpha"],
    )


def weights_from_tensors(tensors: dict[str, torch.Tensor]) -> DiTStageWeights:
    block_count = int(tensors["block_count"].item())
    blocks = []
    for index in range(block_count):
        prefix = f"blocks.{index}"
        blocks.append(
            DiTBlockWeights(
                image=_attention(tensors, f"{prefix}.image"),
                text=_attention(tensors, f"{prefix}.text"),
                image_ffn=_ffn(tensors, f"{prefix}.image_ffn"),
                text_ffn=_ffn(tensors, f"{prefix}.text_ffn"),
                image_modulation=_modulation(tensors, f"{prefix}.image_modulation"),
                text_modulation=_modulation(tensors, f"{prefix}.text_modulation"),
                epsilon=float(tensors[f"{prefix}.epsilon"].item()),
            )
        )
    return DiTStageWeights(
        blocks=tuple(blocks),
        input=tensors["input"],
        output=tensors["output"],
        timesteps=tuple(float(value) for value in tensors["timesteps"].tolist()),
        query_heads=int(tensors["query_heads"].item()),
        head_dimension=int(tensors["head_dimension"].item()),
        latent_positions=int(tensors["latent_positions"].item()),
    )


def _put_attention(output: dict[str, torch.Tensor], prefix: str, weights: AttentionWeights) -> None:
    for name, value in zip(
        ("q", "k", "v", "o"), (weights.q, weights.k, weights.v, weights.o), strict=True
    ):
        output[f"{prefix}.{name}"] = value


def _put_ffn(output: dict[str, torch.Tensor], prefix: str, weights: FeedForwardWeights) -> None:
    output[f"{prefix}.gate"] = weights.gate
    output[f"{prefix}.up"] = weights.up
    output[f"{prefix}.down"] = weights.down


def _put_modulation(
    output: dict[str, torch.Tensor], prefix: str, weights: ModulationWeights
) -> None:
    output[f"{prefix}.gamma"] = weights.gamma
    output[f"{prefix}.beta"] = weights.beta
    output[f"{prefix}.alpha"] = weights.alpha


def weights_to_tensors(weights: DiTStageWeights) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {
        "block_count": torch.tensor(len(weights.blocks)),
        "input": weights.input,
        "output": weights.output,
        "timesteps": torch.tensor(weights.timesteps, dtype=torch.float64),
        "query_heads": torch.tensor(weights.query_heads),
        "head_dimension": torch.tensor(weights.head_dimension),
        "latent_positions": torch.tensor(weights.latent_positions),
    }
    for index, block in enumerate(weights.blocks):
        prefix = f"blocks.{index}"
        _put_attention(output, f"{prefix}.image", block.image)
        _put_attention(output, f"{prefix}.text", block.text)
        _put_ffn(output, f"{prefix}.image_ffn", block.image_ffn)
        _put_ffn(output, f"{prefix}.text_ffn", block.text_ffn)
        _put_modulation(output, f"{prefix}.image_modulation", block.image_modulation)
        _put_modulation(output, f"{prefix}.text_modulation", block.text_modulation)
        output[f"{prefix}.epsilon"] = torch.tensor(block.epsilon, dtype=torch.float64)
    return output


def _compile_ffn(
    weights: FeedForwardWeights,
    basis: torch.Tensor,
    *,
    seed: int,
) -> FeedForwardWeights:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(weights.gate.shape[0], generator=generator)
    scales = torch.empty(weights.gate.shape[0], dtype=weights.gate.dtype)
    scales.uniform_(0.5, 2.0, generator=generator)
    return conjugate_feedforward(weights, basis, basis, permutation, scales)


def _compile_modulation(weights: ModulationWeights, basis: SignedPermutation) -> ModulationWeights:
    return ModulationWeights(
        gamma=weights.gamma[..., basis.permutation],
        beta=basis.apply(weights.beta),
        alpha=weights.alpha[..., basis.permutation],
    )


class DitCompiler:
    def compile(
        self,
        weights: DiTStageWeights,
        output_dir: Path,
        *,
        boundary_key: bytes,
        recipient: StageRecipient,
        seed: int,
    ) -> DitCompilationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        latent_basis = sample_orthogonal(weights.input.shape[1], seed=seed + 17)
        latent_permutation = torch.randperm(
            weights.latent_positions,
            generator=torch.Generator().manual_seed(seed + 19),
        )
        blocks = []
        image_bases = []
        text_bases = []
        for index, block in enumerate(weights.blocks):
            image_signed = SignedPermutation.generate(
                weights.input.shape[0], seed=seed + 101 * index + 23
            )
            text_signed = SignedPermutation.generate(
                weights.input.shape[0], seed=seed + 101 * index + 29
            )
            image_basis = _signed_matrix(image_signed)
            text_basis = _signed_matrix(text_signed)
            image_bases.append(image_basis)
            text_bases.append(text_basis)
            dimension = weights.query_heads * weights.head_dimension
            qk = sample_orthogonal(dimension, seed=seed + 101 * index + 31)
            value = sample_orthogonal(dimension, seed=seed + 101 * index + 37)
            blocks.append(
                DiTBlockWeights(
                    image=conjugate_attention(
                        block.image,
                        image_basis,
                        image_basis,
                        qk,
                        qk,
                        value,
                        torch.linalg.inv(value),
                    ),
                    text=conjugate_attention(
                        block.text,
                        text_basis,
                        text_basis,
                        qk,
                        qk,
                        value,
                        torch.linalg.inv(value),
                    ),
                    image_ffn=_compile_ffn(
                        block.image_ffn, image_basis, seed=seed + 101 * index + 41
                    ),
                    text_ffn=_compile_ffn(block.text_ffn, text_basis, seed=seed + 101 * index + 43),
                    image_modulation=_compile_modulation(block.image_modulation, image_signed),
                    text_modulation=_compile_modulation(block.text_modulation, text_signed),
                    epsilon=block.epsilon,
                )
            )
        transformed = DiTStageWeights(
            blocks=tuple(blocks),
            input=image_bases[0] @ weights.input @ latent_basis.T,
            output=latent_basis @ weights.output @ image_bases[-1].T,
            timesteps=weights.timesteps,
            query_heads=weights.query_heads,
            head_dimension=weights.head_dimension,
            latent_positions=weights.latent_positions,
        )
        stage_tensors = weights_to_tensors(transformed)
        boundary_tensors = {
            "text_basis": text_bases[0],
            "latent_basis": latent_basis,
            "latent_permutation": latent_permutation,
        }
        fingerprint = hashlib.sha256(weights.input.detach().cpu().numpy().tobytes()).hexdigest()
        stage_path = output_dir / "dit-stage.xurcroh"
        boundary_path = output_dir / "dit-client-boundary.xurcroh"
        envelope_path = output_dir / "dit-stage.key.json"
        stage_key = os.urandom(32)
        stage_metadata = write_shard(
            stage_path,
            stage_tensors,
            stage_key,
            {
                "role": "worker-dit-stage",
                "checkpoint_fingerprint": fingerprint,
                "query_heads": weights.query_heads,
                "head_dimension": weights.head_dimension,
            },
        )
        boundary_metadata = write_shard(
            boundary_path,
            boundary_tensors,
            boundary_key,
            {"role": "client-dit-boundary", "checkpoint_fingerprint": fingerprint},
        )
        manifest = {
            "format": "xurcroh-dit-manifest",
            "version": 1,
            "stage_count": 1,
            "checkpoint_fingerprint": fingerprint,
            "stages": [
                {
                    **stage_metadata,
                    "file": stage_path.name,
                    "key_envelope": envelope_path.name,
                    "key_id": recipient.key_id,
                    "recipient_public": recipient.public_key.hex(),
                }
            ],
            "boundary": {**boundary_metadata, "file": boundary_path.name},
        }
        digest = manifest_digest(manifest)
        manifest["digest"] = digest
        envelope = seal_stage_key(
            stage_key,
            recipient,
            manifest_digest=digest,
            checkpoint_fingerprint=fingerprint,
        )
        envelope_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        manifest_path = output_dir / "dit-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return DitCompilationResult(manifest_path, stage_path, boundary_path, envelope_path)
