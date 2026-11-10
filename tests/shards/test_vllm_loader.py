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
