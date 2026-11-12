from __future__ import annotations

import ctypes
from pathlib import Path


class FortressNativeController:
    def __init__(self, library: str | Path) -> None:
        self._library = ctypes.CDLL(str(library))
        self._u32 = ctypes.POINTER(ctypes.c_uint32)
        self._u8 = ctypes.POINTER(ctypes.c_uint8)
        self._stream = ctypes.c_void_p(0)

    def _words(self, values: list[int]) -> ctypes.Array[ctypes.c_uint32]:
        return (ctypes.c_uint32 * len(values))(*values)

    def matvec(self, weights: list[list[int]], activations: list[int]) -> list[int]:
        if not weights or not weights[0] or len(weights) != len(activations):
            raise ValueError("matrix and vector dimensions do not match")
        rows = len(weights)
        cols = len(weights[0])
        if any(len(row) != cols for row in weights):
            raise ValueError("matrix rows must have equal length")
        flat = [value for row in weights for value in row]
        weight_buffer = self._words(flat)
        activation_buffer = self._words(activations)
        output = self._words([0] * cols)
        result = self._library.xurcroh_fortress_matvec_u32(
            weight_buffer,
            activation_buffer,
            output,
            rows,
            cols,
            self._stream,
        )
        if result != 0:
            raise RuntimeError(f"Fortress CUDA matvec failed with status {result}")
        return list(output)

    def chacha_words(self, key: list[int], nonce: bytes, count: int) -> list[int]:
        if len(key) != 8 or len(nonce) != 12 or count < 1:
            raise ValueError("ChaCha requires eight key words, a 12-byte nonce, and positive count")
        key_buffer = self._words(key)
        nonce_buffer = (ctypes.c_uint8 * len(nonce))(*nonce)
        output = self._words([0] * count)
        result = self._library.xurcroh_fortress_chacha_words(
            output,
            count,
            key_buffer,
            nonce_buffer,
            self._stream,
        )
        if result != 0:
            raise RuntimeError(f"Fortress CUDA ChaCha expansion failed with status {result}")
        return list(output)
