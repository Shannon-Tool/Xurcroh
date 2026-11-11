from __future__ import annotations

import pytest
import torch

from xurcroh.errors import CompatibilityError
from xurcroh.runtime.cuda_stage import CudaStage


class Stream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class Buffer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.copies = 0
        self.zeroed = False

    def copy_(self, source: Buffer, non_blocking: bool) -> None:
        assert source.device == self.device
        assert non_blocking
        self.copies += 1

    def zero_(self) -> None:
        self.zeroed = True


def test_cuda_stage_transport_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = Stream()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(
        torch, "empty_like", lambda tensor, device, pin_memory: Buffer(tensor.device)
    )
    stage = CudaStage(0)
    hidden = Buffer(stage.device)
    staged = stage.stage_for_transport(hidden)
    assert staged.copies == 1
    assert stream.synchronizations == 1
    with pytest.raises(ValueError, match="configured device"):
        stage.stage_for_transport(Buffer(torch.device("cpu")))
    stage.close()
    assert staged.zeroed
    assert stage._buffers == []
    stage.close()
    with pytest.raises(RuntimeError, match="closed"):
        stage.stage_for_transport(hidden)


def test_cuda_stage_rejects_unavailable_or_invalid_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CompatibilityError, match="requires a CUDA device"):
        CudaStage(0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(CompatibilityError, match="out of range"):
        CudaStage(-1)
    with pytest.raises(CompatibilityError, match="out of range"):
        CudaStage(1)
