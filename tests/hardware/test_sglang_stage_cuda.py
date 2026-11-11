from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shards.identities import StageIdentities
from xurcroh.adapters.sglang_stage import create_sglang_stage, tensors_from_proxy

pytestmark = pytest.mark.hardware


def test_sglang_executes_private_stage_on_cuda(tiny_llama_path: Path, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device and sglang==0.5.23 are required")
    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=461)
    stage = create_sglang_stage(
        compiled.manifest_path,
        0,
        identities.private_keys[0],
        checkpoint=tiny_llama_path,
        device=0,
    )
    try:
        hidden = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        output = stage.execute(hidden, "cuda-sglang-stage")
        tensors = tensors_from_proxy(output)
        assert tensors["hidden_states"].is_cuda
        assert tensors["hidden_states"].shape == hidden.shape
        stage.drop_request("cuda-sglang-stage")
    finally:
        stage.close()
