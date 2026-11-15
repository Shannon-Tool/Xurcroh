from __future__ import annotations

import sys
from pathlib import Path

from tests.integration.test_ssh_executor import ssh_testbed

from Xurcroh import Xurcroh
from xurcroh.config import WorkerConfig


def _cpu_workers(base: WorkerConfig) -> list[WorkerConfig]:
    return [
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
        for index in range(4)
    ]


def test_sdk_managed_fortress_inference_over_real_ssh(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with ssh_testbed(tmp_path) as base:
        workers = _cpu_workers(base)
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


def test_sdk_managed_image_generation_over_real_ssh(tmp_path: Path) -> None:
    import pytest

    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    checkpoint = tmp_path / "image-model"
    diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler()).save_pretrained(checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-image": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 731,
                }
            },
        ) as client,
    ):
        response = client.images.generate(
            model="tiny-image",
            prompt="a small red square",
            size="64x64",
            steps=1,
            seed=11,
            timeout=120,
        )
        assert response.data[0].b64_json
        assert response.execution.engine == "fortress-diffusion-grid"
        assert len(response.execution.worker_ids) == 4


def test_sdk_managed_moe_inference_over_real_ssh(tiny_llama_path: Path, tmp_path: Path) -> None:
    from tests.fortress.test_official_models import copy_tokenizer
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
    )
    checkpoint = tmp_path / "moe-model"
    Qwen3MoeForCausalLM(config).save_pretrained(checkpoint, safe_serialization=True)
    copy_tokenizer(tiny_llama_path, checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-moe": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 731,
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny-moe",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        assert len(response.execution.worker_ids) == 4


def test_managed_role_crash_restarts_and_resumes_inference(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                }
            },
        ) as client,
    ):
        request = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 1,
            "timeout": 120,
        }
        first = client.chat.completions.create(**request)
        assert first.usage.completion_tokens == 1
        deployment = client._runtime._deployment
        assert deployment is not None
        role = deployment._roles[0]
        pid = Path(deployment._remote_root) / "model-0" / role.worker.name / "role.pid"
        deployment._executor.run(role.worker, f"kill -KILL $(cat {pid})")
        second = client.chat.completions.create(**request)
        assert second.usage.completion_tokens == 1
        assert client.metrics()["fortress_reconnections"] > 0
