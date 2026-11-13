from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from xurcroh.fortress.arithmetic import FIELD_MODULUS


def canonical_field_words(words: torch.Tensor) -> torch.Tensor:
    return words.to(torch.int64).remainder(2**32).remainder(FIELD_MODULUS)


class FortressNativeController:
    def __init__(self, library: str | Path) -> None:
        self._library = ctypes.CDLL(str(library))
        self._u32 = ctypes.POINTER(ctypes.c_uint32)
        self._u8 = ctypes.POINTER(ctypes.c_uint8)
        self._stream = ctypes.c_void_p(0)
        self._library.xurcroh_fortress_matvec_u32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

    def _device_pointer(self, tensor: torch.Tensor, dtype: torch.dtype) -> ctypes.c_void_p:
        if tensor.dtype != dtype or tensor.device.type != "cuda" or not tensor.is_contiguous():
            raise ValueError(
                "Fortress native inputs must be contiguous CUDA tensors of the expected dtype"
            )
        return ctypes.c_void_p(tensor.data_ptr())

    def matvec_device(
        self, weights: torch.Tensor, activations: torch.Tensor, output: torch.Tensor
    ) -> None:
        if weights.ndim != 2 or activations.ndim != 1 or output.ndim != 1:
            raise ValueError("Fortress CUDA matvec requires a matrix and two vectors")
        rows, cols = weights.shape
        if rows < 1 or cols < 1 or activations.shape[0] != rows or output.shape[0] != cols:
            raise ValueError("matrix and vector dimensions do not match")
        result = self._library.xurcroh_fortress_matvec_u32(
            self._device_pointer(weights, torch.int32),
            self._device_pointer(activations, torch.int32),
            self._device_pointer(output, torch.int32),
            rows,
            cols,
            self._stream,
        )
        if result != 0:
            raise RuntimeError(f"Fortress CUDA matvec failed with status {result}")

    def chacha_device(self, output: torch.Tensor, key: torch.Tensor, nonce: torch.Tensor) -> None:
        if output.ndim != 1 or key.shape != (8,) or nonce.shape != (12,) or output.numel() < 1:
            raise ValueError("ChaCha requires output, eight key words, and a 12-byte nonce")
        result = self._library.xurcroh_fortress_chacha_words(
            self._device_pointer(output, torch.int32),
            output.numel(),
            self._device_pointer(key, torch.int32),
            self._device_pointer(nonce, torch.uint8),
            self._stream,
        )
        if result != 0:
            raise RuntimeError(f"Fortress CUDA ChaCha expansion failed with status {result}")

    def _cuda_words(self, values: list[int]) -> torch.Tensor:
        return (
            torch.tensor(values, dtype=torch.int64)
            .remainder(2**32)
            .to(torch.int32)
            .cuda()
            .contiguous()
        )

    def matvec(self, weights: list[list[int]], activations: list[int]) -> list[int]:
        if not weights or not weights[0] or len(weights) != len(activations):
            raise ValueError("matrix and vector dimensions do not match")
        rows = len(weights)
        cols = len(weights[0])
        if any(len(row) != cols for row in weights):
            raise ValueError("matrix rows must have equal length")
        weight_device = self._cuda_words([value for row in weights for value in row]).reshape(
            rows, cols
        )
        activation_device = self._cuda_words(activations)
        output = torch.empty(cols, dtype=torch.int32, device="cuda")
        try:
            self.matvec_device(weight_device, activation_device, output)
            return canonical_field_words(output.cpu()).tolist()
        finally:
            weight_device.zero_()
            activation_device.zero_()
            output.zero_()

    def chacha_words(self, key: list[int], nonce: bytes, count: int) -> list[int]:
        if len(key) != 8 or len(nonce) != 12 or count < 1:
            raise ValueError("ChaCha requires eight key words, a 12-byte nonce, and positive count")
        key_device = self._cuda_words(key)
        nonce_device = torch.tensor(list(nonce), dtype=torch.uint8, device="cuda")
        output = torch.empty(count, dtype=torch.int32, device="cuda")
        try:
            self.chacha_device(output, key_device, nonce_device)
            return output.cpu().to(torch.int64).remainder(2**32).tolist()
        finally:
            key_device.zero_()
            nonce_device.zero_()
            output.zero_()
