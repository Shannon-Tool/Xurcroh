from __future__ import annotations

import threading

import torch

from xurcroh.shards.memory import secure_zero


class KVCachePool:
    def __init__(self, *, layer_count: int, block_count: int, block_size: int) -> None:
        if layer_count < 1 or block_count < 1 or block_size < 1:
            raise ValueError("KV cache dimensions must be positive")
        self.kv = torch.zeros(layer_count, block_count, block_size, dtype=torch.int64)
        self._owners: dict[str, set[int]] = {}
        self._lock = threading.RLock()

    def write(self, request_id: str, block_ids: list[int], value: int) -> None:
        with self._lock:
            for block_id in block_ids:
                if block_id < 0 or block_id >= self.kv.shape[1]:
                    raise IndexError("KV block index is outside the pool")
                if any(block_id in blocks for blocks in self._owners.values()):
                    raise PermissionError("KV block is already owned by another request")
            self._owners.setdefault(request_id, set()).update(block_ids)
            for block_id in block_ids:
                self.kv[:, block_id].fill_(value)

    def free(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._owners:
                raise PermissionError("only the owning request can free KV blocks")
            block_ids = self._owners.pop(request_id)
            for block_id in block_ids:
                self.kv[:, block_id].zero_()


class RequestMemory:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.closed = False
        self._tensors: list[torch.Tensor] = []
        self._buffers: list[bytearray] = []
        self._lock = threading.Lock()

    def track_tensor(self, tensor: torch.Tensor) -> None:
        with self._lock:
            if self.closed:
                raise RuntimeError("request memory is closed")
            self._tensors.append(tensor)

    def track_buffer(self, buffer: bytearray) -> None:
        with self._lock:
            if self.closed:
                raise RuntimeError("request memory is closed")
            self._buffers.append(buffer)

    def burn(self) -> None:
        with self._lock:
            if self.closed:
                return
            for tensor in self._tensors:
                tensor.zero_()
            for buffer in self._buffers:
                secure_zero(buffer)
            self._tensors.clear()
            self._buffers.clear()
            self.closed = True

    def telemetry(self) -> dict[str, int]:
        return {
            "tensors_scrubbed": int(self.closed),
            "buffers_scrubbed": int(self.closed),
        }
