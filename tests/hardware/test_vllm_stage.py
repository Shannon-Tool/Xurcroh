from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shards.identities import StageIdentities
from xurcroh.adapters.vllm_v1_stage import create_vllm_stage, hidden_from_intermediate

pytestmark = pytest.mark.hardware


def test_vllm_executes_encrypted_stage_on_cuda(tiny_llama_path: Path, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device and vllm==0.15.0 are required")
    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=443)
    stage = create_vllm_stage(
        compiled.manifest_path,
        0,
        identities.private_keys[0],
        checkpoint=tiny_llama_path,
        device=0,
    )
    try:
        hidden = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        output = stage.execute(hidden, "cuda-stage")
        result = hidden_from_intermediate(output)
        assert result.is_cuda
        assert result.shape == hidden.shape
        stage.drop_request("cuda-stage")
    finally:
        stage.close()
