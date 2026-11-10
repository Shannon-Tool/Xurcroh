from __future__ import annotations

import uuid
from dataclasses import dataclass

from xurcroh.controller.registry import WorkerRecord
from xurcroh.errors import ClusterUnavailableError


@dataclass(frozen=True)
class StageAssignment:
    stage_index: int
    worker: WorkerRecord


@dataclass(frozen=True)
class Circuit:
    circuit_id: str
    model: str
    stages: tuple[StageAssignment, ...]
    tensor_parallel: int
    expert_parallel: int


class TopologyPlanner:
    def build(
        self,
        model: str,
        workers: list[WorkerRecord] | tuple[WorkerRecord, ...],
        *,
        pipeline_parallel: int = 1,
        tensor_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> Circuit:
        candidates = [worker for worker in workers if model in worker.models]
        if not candidates:
            raise ClusterUnavailableError(f"no healthy workers advertise model {model!r}")
        if pipeline_parallel > len(candidates):
            raise ClusterUnavailableError(
                f"pipeline degree {pipeline_parallel} exceeds {len(candidates)} eligible workers"
            )
        if tensor_parallel > 1 or expert_parallel > 1:
            required = max(tensor_parallel, expert_parallel)
            groups = {worker.group for worker in candidates if worker.group is not None}
            if not any(
                sum(worker.group == group for worker in candidates) >= required for group in groups
            ):
                raise ClusterUnavailableError(
                    f"tensor_parallel/expert_parallel degree {required} "
                    "requires one high-bandwidth host group"
                )
        selected = candidates[:pipeline_parallel]
        stages = tuple(
            StageAssignment(stage_index=index, worker=worker)
            for index, worker in enumerate(selected)
        )
        return Circuit(
            circuit_id=f"circ_{uuid.uuid4().hex}",
            model=model,
            stages=stages,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
        )
