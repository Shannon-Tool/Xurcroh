from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xurcroh.memory_native import NativeMemoryController, activate_burn_on_read_allocator

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
    probe = controller.probe_shared_memory()
    assert probe.numel() > 0
    assert torch.count_nonzero(probe).item() == 0


def test_chacha20_mask_round_trip() -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    controller = NativeMemoryController(root / "native/build/xurcroh_burn_alloc.so")
    original = torch.arange(1024, dtype=torch.int32, device="cuda").to(torch.uint16)
    masked = original.clone()
    key = torch.arange(8, dtype=torch.int64, device="cuda").to(torch.uint32)
    nonce = torch.arange(12, dtype=torch.int32, device="cuda").to(torch.uint8)
    controller.mask_words(masked, key, nonce, unmask=False)
    assert not torch.equal(masked, original)
    controller.mask_words(masked, key, nonce, unmask=True)
    torch.testing.assert_close(masked, original)


def test_zero_on_free_and_reallocation() -> None:
    require_cuda()
    if os.environ.get("XURCROH_ALLOCATOR_TEST_CHILD") != "1":
        environment = dict(os.environ)
        environment["XURCROH_ALLOCATOR_TEST_CHILD"] = "1"
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).resolve()}::test_zero_on_free_and_reallocation",
                "-q",
                "--runxfail",
            ],
            env=environment,
            check=True,
        )
        return
    root = Path(__file__).parents[2]
    subprocess.run([str(root / "scripts/build-native.sh")], check=True)  # noqa: S603
    library = root / "native/build/xurcroh_burn_alloc.so"
    activate_burn_on_read_allocator(library)
    controller = NativeMemoryController(library)
    before = controller.telemetry()
    sensitive = torch.full((4096,), 0x5A5A, dtype=torch.int32, device="cuda")
    del sensitive
    torch.cuda.empty_cache()
    after = controller.telemetry()
    assert after["scrub_calls"] > before["scrub_calls"]
    assert after["scrubbed_bytes"] > before["scrubbed_bytes"]
    reallocated = torch.empty((4096,), dtype=torch.int32, device="cuda")
    assert torch.count_nonzero(reallocated).item() == 0


def test_failed_scrub_quarantines_allocation() -> None:
    require_cuda()
    import ctypes

    root = Path(__file__).parents[2]
    library = root / "native/build/xurcroh_burn_alloc.so"
    controller = NativeMemoryController(library)
    native = ctypes.CDLL(str(library))
    native.xurcroh_burn_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    native.xurcroh_burn_malloc.restype = ctypes.c_void_p
    native.xurcroh_burn_free.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ssize_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    native.xurcroh_burn_free.restype = None
    before = controller.telemetry()
    pointer = native.xurcroh_burn_malloc(16, 0, None)
    assert pointer is not None
    native.xurcroh_burn_free(pointer, 16, 99, None)
    after = controller.telemetry()
    assert after["quarantined_allocations"] == before["quarantined_allocations"] + 1
    assert after["quarantined_bytes"] >= before["quarantined_bytes"] + 16
