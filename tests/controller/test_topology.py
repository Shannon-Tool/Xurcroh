from __future__ import annotations

import time

import pytest

from xurcroh.controller.registry import WorkerRecord
from xurcroh.controller.topology import TopologyPlanner
from xurcroh.errors import ClusterUnavailableError


def record(worker_id: str, group: str, memory: float) -> WorkerRecord:
    return WorkerRecord(
        worker_id=worker_id,
        endpoint=f"https://{worker_id}.internal",
        models=("qwen-7b",),
        modalities=("text",),
        engine="vllm",
        group=group,
        gpu_model="H100",
        gpu_memory_gb=memory,
        max_concurrent_requests=4,
        registered_at=time.time(),
        last_heartbeat=time.time(),
    )


def test_topology_assigns_distinct_stages_and_respects_local_parallelism() -> None:
    planner = TopologyPlanner()
    workers = [record("a", "host-a", 80), record("b", "host-b", 40), record("c", "host-c", 80)]
    circuit = planner.build(
        "qwen-7b", workers, pipeline_parallel=2, tensor_parallel=1, expert_parallel=1
    )
    assert [stage.worker.worker_id for stage in circuit.stages] == ["a", "b"]
    assert circuit.circuit_id
    with pytest.raises(ClusterUnavailableError, match="tensor_parallel"):
        planner.build("qwen-7b", workers, pipeline_parallel=1, tensor_parallel=2, expert_parallel=1)


def test_topology_reports_missing_model_capacity() -> None:
    planner = TopologyPlanner()
    with pytest.raises(ClusterUnavailableError, match="pipeline"):
        planner.build("qwen-7b", [record("a", "host-a", 80)], pipeline_parallel=2)
    with pytest.raises(ClusterUnavailableError, match="no healthy workers"):
        planner.build("missing", [record("a", "host-a", 80)])
