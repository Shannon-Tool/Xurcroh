from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from xurcroh.errors import CompatibilityError
from xurcroh.memory_native import NativeMemoryController, activate_burn_on_read_allocator


class NativeFunction:
    def __init__(self, library: Library) -> None:
        self.library = library

    def __call__(self, *args) -> int:
        self.library.calls.append(args)
        return self.library.status


class Library:
    def __init__(self) -> None:
        self.calls = []
        self.status = 0
        self.xurcroh_get_scrubbed_bytes = lambda: 4096
        self.xurcroh_get_scrub_calls = lambda: 3
        self.xurcroh_get_quarantined_bytes = lambda: 8192
        self.xurcroh_get_quarantined_allocations = lambda: 1
        self.xurcroh_launch_shared_scrub = NativeFunction(self)
        self.xurcroh_run_leftover_probe = NativeFunction(self)
        self.xurcroh_mask_words = NativeFunction(self)
        self.xurcroh_uniform_page_touch = NativeFunction(self)


class CudaTensor:
    def __init__(self, dtype: torch.dtype, count: int) -> None:
        self.dtype = dtype
        self.count = count
        self.is_cuda = True
        self.device = torch.device("cuda", 0)

    def numel(self) -> int:
        return self.count

    def data_ptr(self) -> int:
        return 1234


def test_allocator_must_be_installed_before_cuda(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "allocator.so"
    library.write_bytes(b"test")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(CompatibilityError, match="before any CUDA allocation"):
        activate_burn_on_read_allocator(library)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(CompatibilityError, match="unavailable"):
        activate_burn_on_read_allocator(tmp_path / "missing.so")

    installed = []

    class Allocator:
        def __init__(self, path, malloc, free):
            installed.append((path, malloc, free))

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda.memory, "CUDAPluggableAllocator", Allocator)
    monkeypatch.setattr(
        torch.cuda.memory, "change_current_allocator", lambda allocator: installed.append("changed")
    )
    activate_burn_on_read_allocator(library)
    assert installed == [
        (str(library), "xurcroh_burn_malloc", "xurcroh_burn_free"),
        "changed",
    ]


def test_native_controller_telemetry_and_cuda_calls(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "memory.so"
    path.write_bytes(b"test")
    library = Library()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=8),
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=77)
    )
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "cpu"}),
    )
    controller = NativeMemoryController(path)
    assert controller.telemetry() == {
        "scrubbed_bytes": 4096,
        "scrub_calls": 3,
        "quarantined_bytes": 8192,
        "quarantined_allocations": 1,
    }
    controller.scrub_shared_memory()
    probe = controller.probe_shared_memory()
    assert probe.shape == (8, (48 * 1024) // 4)
    assert synchronized == [0, 0]
    tensor = CudaTensor(torch.uint16, 16)
    key = CudaTensor(torch.uint32, 8)
    nonce = CudaTensor(torch.uint8, 12)
    controller.mask_words(tensor, key, nonce, unmask=True)
    controller.uniform_page_touch(tensor, stride=128)
    assert len(library.calls) == 4
    library.status = 9
    with pytest.raises(CompatibilityError, match="CUDA status 9"):
        controller.scrub_shared_memory()
    with pytest.raises(CompatibilityError, match="ChaCha20 mask failed"):
        controller.mask_words(tensor, key, nonce, unmask=False)
    with pytest.raises(CompatibilityError, match="uniform page touch failed"):
        controller.uniform_page_touch(tensor, stride=256)


def test_native_controller_validates_library_and_tensor_contracts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(CompatibilityError, match="unavailable"):
        NativeMemoryController(tmp_path / "missing.so")
    path = tmp_path / "memory.so"
    path.write_bytes(b"test")
    monkeypatch.setattr(ctypes, "CDLL", lambda path: Library())
    controller = NativeMemoryController(path)
    tensor = torch.zeros(4, dtype=torch.uint16)
    key = torch.zeros(8, dtype=torch.uint32)
    nonce = torch.zeros(12, dtype=torch.uint8)
    with pytest.raises(TypeError, match="mask requires"):
        controller.mask_words(tensor.float(), key, nonce, unmask=False)
    with pytest.raises(ValueError, match="same CUDA device"):
        controller.mask_words(tensor, key, nonce, unmask=False)
    with pytest.raises(ValueError, match="eight key words"):
        controller.mask_words(
            CudaTensor(torch.uint16, 4),
            CudaTensor(torch.uint32, 7),
            CudaTensor(torch.uint8, 12),
            unmask=False,
        )
    with pytest.raises(ValueError, match="CUDA pages"):
        controller.uniform_page_touch(tensor, stride=128)
    with pytest.raises(ValueError, match="stride of at least 128"):
        controller.uniform_page_touch(CudaTensor(torch.uint8, 128), stride=64)
