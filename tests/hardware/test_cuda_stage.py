from __future__ import annotations

import pytest
import torch

from xurcroh.runtime.cuda_stage import CudaStage

pytestmark = pytest.mark.hardware


def test_cuda_stage_streams_bf16_and_zeroizes_buffers() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for native stage verification")
    stage = CudaStage(0)
    hidden = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")
    staged = stage.stage_for_transport(hidden)
    assert staged.is_pinned()
    torch.testing.assert_close(staged, hidden.cpu())
    stage.zeroize(staged, hidden)
    assert not staged.any()
    assert not hidden.any()
    stage.close()
