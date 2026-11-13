from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from xurcroh.errors import CompatibilityError


def activate_burn_on_read_allocator(library_path: str | Path) -> None:
    if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        raise CompatibilityError("burn allocator must be activated before any CUDA allocation")
    path = Path(library_path)
    if not path.is_file():
        raise CompatibilityError(f"burn allocator library is unavailable: {path}")
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(path), "xurcroh_burn_malloc", "xurcroh_burn_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)


class NativeMemoryController:
    def __init__(self, library_path: str | Path) -> None:
        path = Path(library_path)
        if not path.is_file():
            raise CompatibilityError(f"native memory library is unavailable: {path}")
        self.library = ctypes.CDLL(str(path))
        self.library.xurcroh_get_scrubbed_bytes.restype = ctypes.c_uint64
        self.library.xurcroh_get_scrub_calls.restype = ctypes.c_uint64
        self.library.xurcroh_get_quarantined_bytes.restype = ctypes.c_uint64
        self.library.xurcroh_get_quarantined_allocations.restype = ctypes.c_uint64
        self.library.xurcroh_launch_shared_scrub.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.library.xurcroh_run_leftover_probe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.xurcroh_mask_words.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]
        self.library.xurcroh_uniform_page_touch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

    def telemetry(self) -> dict[str, int]:
        return {
            "scrubbed_bytes": self.library.xurcroh_get_scrubbed_bytes(),
            "scrub_calls": self.library.xurcroh_get_scrub_calls(),
            "quarantined_bytes": self.library.xurcroh_get_quarantined_bytes(),
            "quarantined_allocations": self.library.xurcroh_get_quarantined_allocations(),
        }

    def scrub_shared_memory(self, device: int = 0) -> None:
        properties = torch.cuda.get_device_properties(device)
        stream = torch.cuda.current_stream(device).cuda_stream
        status = self.library.xurcroh_launch_shared_scrub(
            properties.multi_processor_count, ctypes.c_void_p(stream)
        )
        if status != 0:
            raise CompatibilityError(f"shared-memory scrub failed with CUDA status {status}")
        torch.cuda.synchronize(device)

    def probe_shared_memory(self, device: int = 0) -> torch.Tensor:
        properties = torch.cuda.get_device_properties(device)
        words = (48 * 1024) // 4
        output = torch.empty(
            (properties.multi_processor_count, words), dtype=torch.int32, device=device
        )
        stream = torch.cuda.current_stream(device).cuda_stream
        status = self.library.xurcroh_run_leftover_probe(
            ctypes.c_void_p(output.data_ptr()),
            properties.multi_processor_count,
            ctypes.c_void_p(stream),
        )
        if status != 0:
            raise CompatibilityError(f"shared-memory probe failed with CUDA status {status}")
        torch.cuda.synchronize(device)
        return output

    def mask_words(
        self, tensor: torch.Tensor, key: torch.Tensor, nonce: torch.Tensor, *, unmask: bool
    ) -> None:
        if tensor.dtype != torch.uint16 or key.dtype != torch.uint32 or nonce.dtype != torch.uint8:
            raise TypeError("mask requires uint16 tensor, uint32 key, and uint8 nonce")
        if not tensor.is_cuda or not key.is_cuda or not nonce.is_cuda:
            raise ValueError("mask tensors must reside on the same CUDA device")
        if key.numel() != 8 or nonce.numel() != 12:
            raise ValueError("ChaCha20 requires eight key words and a 96-bit nonce")
        status = self.library.xurcroh_mask_words(
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            ctypes.c_void_p(key.data_ptr()),
            ctypes.c_void_p(nonce.data_ptr()),
            unmask,
            ctypes.c_void_p(torch.cuda.current_stream(tensor.device).cuda_stream),
        )
        if status != 0:
            raise CompatibilityError(f"ChaCha20 mask failed with CUDA status {status}")

    def uniform_page_touch(self, pages: torch.Tensor, *, stride: int) -> None:
        if not pages.is_cuda or stride < 128:
            raise ValueError(
                "uniform page touch requires CUDA pages and a stride of at least 128 bytes"
            )
        status = self.library.xurcroh_uniform_page_touch(
            ctypes.c_void_p(pages.data_ptr()),
            pages.numel(),
            stride,
            ctypes.c_void_p(torch.cuda.current_stream(pages.device).cuda_stream),
        )
        if status != 0:
            raise CompatibilityError(f"uniform page touch failed with CUDA status {status}")
