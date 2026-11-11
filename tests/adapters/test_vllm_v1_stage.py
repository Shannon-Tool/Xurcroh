from __future__ import annotations

import sys
import types

import pytest
import torch

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.adapters.vllm_v1_stage import (
    VllmStageRunner,
    decode_intermediate,
    encode_intermediate,
    require_vllm_v1,
)
from xurcroh.errors import CompatibilityError


class Intermediate:
    def __init__(self, tensors):
        self.tensors = tensors


def install_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class SamplingParams:
        def __init__(self, temperature: float) -> None:
            self.temperature = temperature

    class NewRequestData:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class CachedRequestData:
        @classmethod
        def make_empty(cls):
            return cls()

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class SchedulerOutput:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    output = types.ModuleType("vllm.v1.core.sched.output")
    output.NewRequestData = NewRequestData
    output.CachedRequestData = CachedRequestData
    output.SchedulerOutput = SchedulerOutput
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.SamplingParams = SamplingParams
    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.GroupCoordinator = type("GroupCoordinator", (), {"is_last_rank": True})
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = Intermediate
    for name, module in {
        "vllm.sampling_params": sampling,
        "vllm.v1.core.sched.output": output,
        "vllm.distributed.parallel_state": parallel,
        "vllm.sequence": sequence,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_vllm_scheduler_and_intermediate_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    install_vllm_stubs(monkeypatch)
    monkeypatch.setattr("xurcroh.adapters.vllm_v1_stage.require_vllm_v1", lambda: None)

    class Runner:
        def execute_model(self, scheduler, intermediate):
            hidden_states = (
                scheduler.scheduled_new_reqs[0].prompt_embeds[0]
                if scheduler.scheduled_new_reqs
                else intermediate.tensors["hidden_states"]
            )
            return Intermediate({"hidden_states": hidden_states, "residual": torch.ones(1, 4)})

    runner = VllmStageRunner(Runner())
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    output = runner.execute(hidden, "request", 0)
    assert output.tensors["hidden_states"].shape == (2, 4)
    assert output.tensors["residual"].shape == (1, 4)

    key = bytes(range(32))
    sender = PipelineTensorInterceptor(key, b"client", b"stage", b"vllm-test")
    receiver = PipelineTensorInterceptor(key, b"stage", b"client", b"vllm-test")
    packet, shape = encode_intermediate(output.tensors, "request", sender)
    decoded = decode_intermediate(packet, receiver)
    assert shape[0] > 0
    torch.testing.assert_close(decoded["hidden_states"], output.tensors["hidden_states"])
    torch.testing.assert_close(decoded["residual"], output.tensors["residual"])


def test_vllm_version_is_required() -> None:
    with pytest.raises(CompatibilityError, match=r"vllm==0\.15\.0"):
        require_vllm_v1()


def test_vllm_request_drop_removes_input_batch_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    install_vllm_stubs(monkeypatch)
    removed = []
    runner = VllmStageRunner(
        type(
            "Runner",
            (),
            {
                "input_batch": type(
                    "Batch",
                    (),
                    {"remove_request": lambda self, request_id: removed.append(request_id)},
                )()
            },
        )()
    )
    runner.drop_request("request")
    assert removed == ["request"]


def test_strict_reference_engine_is_rejected(tiny_llama_path, tmp_path) -> None:
    from shards.identities import StageIdentities
    from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
    from xurcroh.runtime.stage import DenseStageEngine

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=439)
    with pytest.raises(CompatibilityError, match="reference engine"):
        DenseStageEngine(
            "tiny",
            ModelDefinition(
                path=str(tiny_llama_path),
                manifest_path=str(compiled.manifest_path),
                engine="reference",
            ),
            [WorkerConfig(name="worker", endpoint="https://worker.invalid")],
            SecurityConfig(mode="strict"),
        )


def test_vllm_stage_lifecycle_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from xurcroh.adapters.vllm_v1_stage import VllmStage, hidden_from_intermediate
    from xurcroh.errors import ModelLoadError, ProtocolError

    install_vllm_stubs(monkeypatch)
    monkeypatch.setattr("xurcroh.adapters.vllm_v1_stage.require_vllm_v1", lambda: None)
    removed = []
    calls = []

    class Runner:
        input_batch = type(
            "Batch", (), {"remove_request": lambda self, request_id: removed.append(request_id)}
        )()

        def execute_model(self, scheduler, intermediate):
            calls.append((scheduler, intermediate))
            return Intermediate({"hidden_states": torch.ones(1, 4), "residual": torch.ones(1, 4)})

    worker = type("Worker", (), {"shutdown": lambda self: removed.append("shutdown")})()
    stage = VllmStage(Runner(), worker)
    first = stage.execute(torch.ones(2, 4), "request")
    second = stage.execute(first, "request")
    assert stage._positions["request"] == 3
    assert calls[1][0].scheduled_cached_reqs.num_computed_tokens == [2]
    assert hidden_from_intermediate(second).shape == (1, 4)
    stage.drop_request("request")
    assert "request" in removed
    with pytest.raises(ProtocolError):
        stage.execute(second, None)
    stage.close()
    assert stage.closed
    with pytest.raises(ModelLoadError):
        stage.execute(second, "request")


def test_vllm_stage_rejects_out_of_range_index(tiny_llama_path, tmp_path, monkeypatch) -> None:
    from shards.identities import StageIdentities
    from xurcroh.adapters.vllm_v1_stage import create_vllm_stage
    from xurcroh.errors import ModelLoadError

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=467)
    monkeypatch.setattr("xurcroh.adapters.vllm_v1_stage.require_vllm_v1", lambda: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    with pytest.raises(ModelLoadError, match="out of range"):
        create_vllm_stage(
            compiled.manifest_path,
            1,
            identities.private_keys[0],
            checkpoint=tiny_llama_path,
        )


def test_vllm_version_rejects_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.14.0")
    with pytest.raises(CompatibilityError, match="unsupported vLLM version"):
        require_vllm_v1()
