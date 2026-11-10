from __future__ import annotations

import json
from pathlib import Path

import torch

from shards.identities import StageIdentities
from xurcroh.algebra.attention import AttentionWeights
from xurcroh.algebra.feedforward import FeedForwardWeights
from xurcroh.models.dit import (
    DiTBlockWeights,
    DiTStageWeights,
    ModulationWeights,
    reference_dit_stage,
)
from xurcroh.shards.dit_compiler import DitCompiler, weights_from_tensors
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key


def tiny_dit_weights() -> DiTStageWeights:
    generator = torch.Generator().manual_seed(401)

    def matrix(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64) / 4

    def attention() -> AttentionWeights:
        return AttentionWeights(q=matrix(4, 4), k=matrix(4, 4), v=matrix(4, 4), o=matrix(4, 4))

    def ffn() -> FeedForwardWeights:
        return FeedForwardWeights(gate=matrix(5, 4), up=matrix(5, 4), down=matrix(4, 5))

    def modulation() -> ModulationWeights:
        return ModulationWeights(gamma=matrix(4), beta=matrix(4), alpha=matrix(4))

    block = DiTBlockWeights(
        image=attention(),
        text=attention(),
        image_ffn=ffn(),
        text_ffn=ffn(),
        image_modulation=modulation(),
        text_modulation=modulation(),
    )
    return DiTStageWeights(
        blocks=(block,),
        input=matrix(4, 2),
        output=matrix(2, 4),
        timesteps=(1.0, 0.5, 0.0),
        query_heads=1,
        head_dimension=4,
        latent_positions=3,
    )


def test_compiled_dit_matches_original_without_worker_transformation_secrets(
    tmp_path: Path,
) -> None:
    original = tiny_dit_weights()
    identities = StageIdentities.create(1)
    result = DitCompiler().compile(
        original,
        tmp_path / "compiled",
        boundary_key=identities.boundary_key,
        recipient=identities.recipients[0],
        seed=409,
    )
    manifest = json.loads(result.manifest_path.read_text())
    envelope = json.loads(result.key_envelope.read_text())
    stage_key = open_stage_key(
        envelope,
        identities.private_keys[0],
        manifest_digest=manifest["digest"],
        checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
    )
    tensors, _ = read_shard(result.stage_shard, stage_key)
    assert not any("basis" in name or "permutation" in name for name in tensors)
    transformed = weights_from_tensors(tensors)
    boundary, _ = read_shard(result.boundary_shard, identities.boundary_key)

    latent = torch.randn(3, 2, dtype=torch.float64)
    text = torch.randn(2, 4, dtype=torch.float64)
    protected_latent = latent[boundary["latent_permutation"]] @ boundary["latent_basis"].T
    protected_text = text @ boundary["text_basis"].T
    actual = reference_dit_stage(transformed, protected_latent, protected_text, steps=2)
    recovered = (actual @ boundary["latent_basis"])[torch.argsort(boundary["latent_permutation"])]
    expected = reference_dit_stage(original, latent, text, steps=2)
    torch.testing.assert_close(recovered, expected, atol=3e-10, rtol=3e-10)
