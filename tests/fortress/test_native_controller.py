from __future__ import annotations

from types import SimpleNamespace

import torch

from xurcroh.fortress.native import FortressNativeController


class FakeDeviceTensor:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = torch.int32
        self.device = SimpleNamespace(type="cuda")
        self._pointer = pointer

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return self._pointer


def test_native_controller_passes_cuda_device_pointers(tmp_path) -> None:
    calls = []

    class Library:
        def xurcroh_fortress_matvec_u32(self, *arguments) -> int:
            calls.append(arguments)
            return 0

    import ctypes

    library = tmp_path / "libfortress.so"
    library.write_bytes(b"")
    controller = FortressNativeController.__new__(FortressNativeController)
    controller._library = Library()
    controller._stream = ctypes.c_void_p(0)
    weights = FakeDeviceTensor((3, 2), 0x1000)
    activations = FakeDeviceTensor((3,), 0x2000)
    output = FakeDeviceTensor((2,), 0x3000)
    controller.matvec_device(weights, activations, output)  # type: ignore[arg-type]
    assert tuple(argument.value for argument in calls[0][:3]) == (0x1000, 0x2000, 0x3000)
    assert calls[0][3:5] == (3, 2)


def test_native_controller_rejects_host_tensors(tmp_path) -> None:
    import ctypes

    library = tmp_path / "libfortress.so"
    library.write_bytes(b"")
    controller = FortressNativeController.__new__(FortressNativeController)
    controller._library = SimpleNamespace(xurcroh_fortress_matvec_u32=lambda *arguments: 0)
    controller._stream = ctypes.c_void_p(0)
    host = torch.zeros(2, 2, dtype=torch.int32)
    vector = torch.zeros(2, dtype=torch.int32)
    with __import__("pytest").raises(ValueError, match="CUDA"):
        controller.matvec_device(host, vector, vector)
