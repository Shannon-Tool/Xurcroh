"""Decoy tasks and the cover mode are engine settings that every trace reports."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tests.fortress.test_sdk import _signed_remote_grid

from Xurcroh import Xurcroh
from xurcroh.errors import XurcrohError


def _client(tiny_llama_path: Path, cluster: list[dict]) -> Xurcroh:
    return Xurcroh(
        cluster=cluster,
        model_path={
            "tiny": {
                "path": str(tiny_llama_path),
                "revision": "a" * 40,
                "engine": "fortress",
                "fortress_seed": 731,
                "fortress_shares": 2,
            }
        },
    )


def _ask(client: Xurcroh) -> dict:
    response = client.chat.completions.create(
        model="tiny",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        max_tokens=1,
    )
    return client.tasks.trace(response.execution.request_id)


def test_decoy_tasks_run_between_requests_and_leave_request_accounting_alone(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, 2)
    for worker in cluster:
        worker["labels"].update({"decoy_interval": "0.05", "decoy_max_tokens": "2"})
    try:
        with _client(tiny_llama_path, cluster) as client:
            first = _ask(client)
            engine = client._runtime._engines["tiny"]
            deadline = time.monotonic() + 60
            while engine.decoy_task_count < 3:
                assert time.monotonic() < deadline, "no decoy task ran"
                time.sleep(0.05)
            second = _ask(client)
            assert engine.decoy_failure_count == 0
    finally:
        for worker in workers:
            worker.close()

    # Each trace states that decoys were on and how often, never which tasks were decoys.
    for trace in (first, second):
        assert trace["transport"]["decoy_tasks"] == {
            "mean_interval_seconds": 0.05,
            "max_tokens": 2,
        }
    # A decoy goes through the same code as a request, and the counters in a trace are
    # those of its own request only: the same request costs the same calls both times.
    assert (
        first["transport"]["counters"]["fortress_role_requests"]
        == second["transport"]["counters"]["fortress_role_requests"]
        > 0
    )


def test_cover_mode_is_applied_and_reported(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, 2)
    for worker in cluster:
        worker["labels"].update({"cover_mode": "constant-rate", "cover_interval": "0.005"})
    try:
        with _client(tiny_llama_path, cluster) as client:
            trace = _ask(client)
            engine = client._runtime._engines["tiny"]
            assert engine._grid_runtime.constant_rate is True
    finally:
        for worker in workers:
            worker.close()
    assert trace["transport"]["cover_mode"] == "constant-rate"
    assert trace["transport"]["cover_interval_seconds"] == 0.005
    assert trace["transport"]["decoy_tasks"] is None


def test_unknown_cover_mode_is_refused(
    tiny_llama_path: Path, tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, 2)
    for worker in cluster:
        worker["labels"]["cover_mode"] = "sometimes"
    try:
        with (
            _client(tiny_llama_path, cluster) as client,
            pytest.raises(XurcrohError, match="cover_mode"),
        ):
            _ask(client)
    finally:
        for worker in workers:
            worker.close()
