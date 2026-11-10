from __future__ import annotations

import torch

from xurcroh.errors import CompatibilityError


class CudaStage:
    def __init__(self, device: int) -> None:
        if not torch.cuda.is_available():
            raise CompatibilityError("CUDA stage execution requires a CUDA device")
        if device < 0 or device >= torch.cuda.device_count():
            raise CompatibilityError("CUDA stage device index is out of range")
        self.device = torch.device("cuda", device)
        self._buffers: list[torch.Tensor] = []
        self._closed = False

    def stage_for_transport(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("CUDA stage is closed")
        if not hidden.is_cuda or hidden.device != self.device:
            raise ValueError("CUDA stage input must reside on the configured device")
        stream = torch.cuda.current_stream(self.device)
        staged = torch.empty_like(hidden, device="cpu", pin_memory=True)
        staged.copy_(hidden, non_blocking=True)
        stream.synchronize()
        self._buffers.append(staged)
        return staged

    def zeroize(self, *tensors: torch.Tensor) -> None:
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.current_stream(self.device).synchronize()

    def close(self) -> None:
        if not self._closed:
            self.zeroize(*self._buffers)
            self._buffers.clear()
            self._closed = True
