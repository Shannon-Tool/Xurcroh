from __future__ import annotations

import torch

from xurcroh.memory import KVCachePool, RequestMemory


def test_kv_pool_zeroes_physical_slots_before_reuse() -> None:
    pool = KVCachePool(layer_count=2, block_count=4, block_size=8)
    pool.write("user-a", [1, 3], 0xDEADBEEF)
    pool.free("user-a")
    assert torch.count_nonzero(pool.kv) == 0
    pool.write("user-b", [1], 42)
    assert pool.kv[:, 1].sum().item() == 42 * 16
    assert pool.kv[:, 3].sum().item() == 0


def test_cross_user_free_is_rejected() -> None:
    import pytest

    pool = KVCachePool(layer_count=1, block_count=2, block_size=4)
    pool.write("user-a", [0], 7)
    with pytest.raises(PermissionError):
        pool.free("user-b")


def test_request_memory_burns_activations_latents_and_buffers() -> None:
    tensor = torch.ones(4)
    buffer = bytearray(b"secret")
    memory = RequestMemory("request-a")
    memory.track_tensor(tensor)
    memory.track_buffer(buffer)
    memory.burn()
    assert torch.count_nonzero(tensor) == 0
    assert buffer == bytes(len(buffer))
    assert memory.closed
    telemetry = memory.telemetry()
    assert telemetry["tensors_scrubbed"] == 1
    assert telemetry["buffers_scrubbed"] == 1
    assert "request-a" not in str(telemetry)
