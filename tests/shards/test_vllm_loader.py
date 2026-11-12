from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from xurcroh.adapters.vllm_loader import XurcrohShardedModelLoader
from xurcroh.shards.format import write_shard


def test_vllm_loader_uses_only_secure_shard(tmp_path: Path) -> None:
    shard = tmp_path / "stage.xurcroh"
    write_shard(shard, {"weight": torch.full((2, 2), 3.0)}, bytes(32), {"stage_index": 0})
    config = SimpleNamespace(
        model_loader_extra_config={
            "shard_path": str(shard),
            "stage_key": bytes(32),
            "expected_metadata": {"stage_index": 0},
        }
    )
    loader = XurcrohShardedModelLoader(config)
    loader.download_model(SimpleNamespace())
    model = torch.nn.Linear(2, 2, bias=False)
    loader.load_weights(model, SimpleNamespace())
    torch.testing.assert_close(model.weight, torch.full((2, 2), 3.0))


def test_vllm_loader_uses_manifest_recipient_identity(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from shards.identities import StageIdentities
    from xurcroh.shards.format import read_shard
    from xurcroh.shards.key_broker import register_stage_identity, revoke_stage_identity

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=449)
    handle = register_stage_identity(identities.private_keys[0])
    config = SimpleNamespace(
        model_loader_extra_config={
            "manifest_path": str(compiled.manifest_path),
            "stage_index": 0,
            "stage_identity_handle": handle,
        }
    )
    try:
        loader = XurcrohShardedModelLoader(config)
        secure = loader._secure_loader()
        tensors, _ = read_shard(secure.path, secure.key)
        assert "model.layers.0.self_attn.q_proj.weight" in tensors
        loader.download_model(SimpleNamespace())
    finally:
        revoke_stage_identity(handle)
