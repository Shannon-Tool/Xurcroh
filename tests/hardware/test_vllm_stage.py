from __future__ import annotations

import json
import os
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
    from xurcroh.stage_worker import DenseStage

    reference = DenseStage(compiled.manifest_path, 0, identities.private_keys[0])
    stage = create_vllm_stage(
        compiled.manifest_path,
        0,
        identities.private_keys[0],
        checkpoint=tiny_llama_path,
        device=0,
    )
    errors = []
    try:
        hidden = torch.randn(2, 16, dtype=torch.float32)
        decode = torch.randn(1, 16, dtype=torch.float32)
        for current in (hidden, decode):
            output = stage.execute(current.cuda(), "cuda-stage")
            result = hidden_from_intermediate(output)
            expected = reference.execute(current.double(), "cuda-stage")
            assert result.is_cuda
            assert result.shape == current.shape
            actual_cpu = result.cpu().double()
            errors.append((actual_cpu - expected).abs().max().item())
            torch.testing.assert_close(actual_cpu, expected, atol=0.05, rtol=0.05)
        stage.drop_request("cuda-stage")
        report_path = os.environ.get("XURCROH_HARDWARE_REPORT")
        if report_path:
            Path(report_path).write_text(
                json.dumps(
                    {
                        "test": "vllm-prefill-and-decode",
                        "status": "passed",
                        "max_abs_errors": errors,
                        "atol": 0.05,
                        "rtol": 0.05,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        stage.close()
        reference.close()
