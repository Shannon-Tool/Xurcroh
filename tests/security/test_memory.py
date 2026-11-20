from __future__ import annotations

from pathlib import Path

import torch

from xurcroh.memory import KVCachePool, RequestMemory


def _compiled_stage(tmp_path: Path, tiny_llama_path: Path):
    from shards.identities import StageIdentities

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=97)
    return identities, compiled


def _assert_zeroed(caches) -> None:
    for cache in caches:
        for key, value in cache:
            assert torch.count_nonzero(key) == 0
            assert torch.count_nonzero(value) == 0


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


def test_session_close_scrubs_cached_kv_tensors(tiny_llama_path: Path, tmp_path: Path) -> None:
    from xurcroh.stage_worker import DenseStage

    identities, compiled = _compiled_stage(tmp_path, tiny_llama_path)
    stage = DenseStage(compiled.manifest_path, 0, identities.private_keys[0])
    hidden = torch.randn(2, 16, dtype=torch.float64)
    stage.execute(hidden, "req-1", state_key="sess-1:req-1")
    assert stage.runner.cache_entries == 1
    cached = stage.runner._caches["sess-1:req-1"]
    stage.drop_session("sess-1")
    assert stage.runner.cache_entries == 0
    _assert_zeroed([cached])


def test_dense_stage_close_scrubs_all_runner_caches(tiny_llama_path: Path, tmp_path: Path) -> None:
    from xurcroh.stage_worker import DenseStage

    identities, compiled = _compiled_stage(tmp_path, tiny_llama_path)
    stage = DenseStage(compiled.manifest_path, 0, identities.private_keys[0])
    hidden = torch.randn(2, 16, dtype=torch.float64)
    stage.execute(hidden, "req-1", state_key="sess-1:req-1")
    stage.execute(hidden, "req-2", state_key="sess-2:req-2")
    assert stage.runner.cache_entries == 2
    cached = list(stage.runner._caches.values())
    stage.close()
    assert stage.runner.cache_entries == 0
    _assert_zeroed(cached)


def test_stage_close_session_route_drops_session_caches(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    import base64

    from fastapi.testclient import TestClient

    from xurcroh.runtime.stage_sessions import ClientStageSession
    from xurcroh.stage_worker import create_stage_app

    identities, compiled = _compiled_stage(tmp_path, tiny_llama_path)
    app = create_stage_app(
        compiled.manifest_path, stage_index=0, private_key=identities.private_keys[0]
    )
    session = ClientStageSession(0, identities.recipients[0].public_key)
    hidden = torch.randn(2, 16, dtype=torch.float64)
    packet = session.sender.seal("req-1", "f64", [2, 16], hidden.numpy().tobytes())
    body = {
        "request_id": "req-1",
        "session_id": session.session_id,
        "session_token": session.session_token,
        "client_public": session.client_public,
        "handshake": base64.b64encode(session.handshake).decode(),
        "stage_index": 0,
        "dtype": "f64",
        "shape": [2, 16],
        "packet": base64.b64encode(packet).decode(),
    }
    with TestClient(app) as client:
        response = client.post("/v1/stages/execute", json=body)
        assert response.status_code == 200
        assert client.get("/v1/capabilities").json()["cache_entries"] == 1
        closed = client.delete(
            f"/v1/stages/sessions/{session.session_id}", headers=session.auth_headers()
        )
        assert closed.status_code == 204
        assert client.get("/v1/capabilities").json()["cache_entries"] == 0
