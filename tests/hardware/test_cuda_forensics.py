from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from xurcroh.memory_native import NativeMemoryController

pytestmark = pytest.mark.hardware


def require_cuda() -> None:
    if shutil.which("nvcc") is None:
        pytest.skip("CUDA compiler and GPU are required for native forensics")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-capable GPU is required for native forensics")


def test_allocator_and_shared_scrubber() -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    subprocess.run([str(root / "scripts/build-native.sh")], check=True)  # noqa: S603
    controller = NativeMemoryController(root / "native/build/xurcroh_burn_alloc.so")
    controller.scrub_shared_memory()
    assert controller.telemetry()["scrub_calls"] >= 0


def test_chacha20_mask_round_trip() -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    controller = NativeMemoryController(root / "native/build/otp_mask_kernels.so")
    original = torch.arange(1024, dtype=torch.uint16, device="cuda")
    masked = original.clone()
    key = torch.arange(8, dtype=torch.uint32, device="cuda")
    nonce = torch.arange(12, dtype=torch.uint8, device="cuda")
    controller.mask_words(masked, key, nonce, unmask=False)
    assert not torch.equal(masked, original)
    controller.mask_words(masked, key, nonce, unmask=True)
    torch.testing.assert_close(masked, original)
