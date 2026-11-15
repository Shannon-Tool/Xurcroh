from __future__ import annotations

import sys
from pathlib import Path

from tests.integration.test_ssh_executor import ssh_testbed

from Xurcroh import Xurcroh
from xurcroh.config import WorkerConfig


def test_sdk_managed_fortress_inference_over_real_ssh(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with ssh_testbed(tmp_path) as base:
        workers = []
        for index in range(4):
            workers.append(
                WorkerConfig(
                    name=f"cpu-{index}",
                    host=base.host,
                    ssh=base.ssh,
                    labels={
                        "device": "cpu",
                        "bootstrap": "false",
                        "python_path": sys.executable,
                    },
                    resources={"max_host_memory_gb": 16},
                )
            )
        with Xurcroh(
            cluster=workers,
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                }
            },
        ) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
                timeout=120,
            )
            assert response.usage.completion_tokens == 1
            assert response.execution.engine == "fortress-grid"
            assert len(response.execution.worker_ids) == 4
            endpoints = client._runtime._workers_for_model("tiny")
            assert [worker.endpoint for worker in endpoints] == [
                f"quic://127.0.0.1:{9444 + index}" for index in range(4)
            ]
