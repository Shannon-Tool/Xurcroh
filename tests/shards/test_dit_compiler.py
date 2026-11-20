from __future__ import annotations

import json
from pathlib import Path

import torch

from models.test_dit_stage import tiny_dit_weights
from shards.identities import StageIdentities
from xurcroh.models.dit import DiTStageWeights
from xurcroh.shards.dit_compiler import DitCompiler


def _compile(weights, output_dir: Path, identities: StageIdentities, **kwargs) -> dict:
    result = DitCompiler().compile(
        weights,
        output_dir,
        boundary_key=identities.boundary_key,
        recipient=identities.recipients[0],
        **kwargs,
    )
    return json.loads(result.manifest_path.read_text())


def test_dit_fingerprint_covers_all_checkpoint_tensors(tmp_path: Path) -> None:
    identities = StageIdentities.create(1)
    original = tiny_dit_weights()
    modified = DiTStageWeights(
        blocks=original.blocks,
        input=original.input,
        output=original.output + 0.125,
        timesteps=original.timesteps,
        query_heads=original.query_heads,
        head_dimension=original.head_dimension,
        latent_positions=original.latent_positions,
    )
    first = _compile(original, tmp_path / "first", identities, seed=409)
    second = _compile(modified, tmp_path / "second", identities, seed=409)
    assert first["checkpoint_fingerprint"] != second["checkpoint_fingerprint"]


def test_dit_fingerprint_is_stable_for_identical_checkpoints(tmp_path: Path) -> None:
    identities = StageIdentities.create(1)
    first = _compile(tiny_dit_weights(), tmp_path / "first", identities, seed=409)
    second = _compile(tiny_dit_weights(), tmp_path / "second", identities, seed=409)
    assert first["checkpoint_fingerprint"] == second["checkpoint_fingerprint"]


def test_dit_seed_defaults_to_csprng_secret(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "xurcroh.shards.dit_compiler.secrets.randbits",
        lambda bits: calls.append(bits) or 409,
    )
    identities = StageIdentities.create(1)
    automatic = DitCompiler().compile(
        tiny_dit_weights(),
        tmp_path / "automatic",
        boundary_key=identities.boundary_key,
        recipient=identities.recipients[0],
    )
    assert calls == [63]
    explicit = DitCompiler().compile(
        tiny_dit_weights(),
        tmp_path / "explicit",
        boundary_key=identities.boundary_key,
        recipient=identities.recipients[0],
        seed=409,
    )
    automatic_stage = _read_stage(automatic, identities)
    explicit_stage = _read_stage(explicit, identities)
    assert automatic_stage.keys() == explicit_stage.keys()
    for name in automatic_stage:
        assert torch.equal(automatic_stage[name], explicit_stage[name])


def _read_stage(result, identities: StageIdentities) -> dict[str, torch.Tensor]:
    from xurcroh.shards.format import read_shard
    from xurcroh.shards.recipients import open_stage_key

    manifest = json.loads(result.manifest_path.read_text())
    envelope = json.loads(result.key_envelope.read_text())
    stage_key = open_stage_key(
        envelope,
        identities.private_keys[0],
        manifest_digest=manifest["digest"],
        checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
    )
    tensors, _ = read_shard(result.stage_shard, stage_key)
    return tensors
