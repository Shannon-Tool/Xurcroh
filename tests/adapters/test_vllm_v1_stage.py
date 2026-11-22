from __future__ import annotations

import base64
import contextlib
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
from xurcroh.errors import CompatibilityError, ProtocolError


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
    parallel.GroupCoordinator = type(
        "GroupCoordinator", (), {"is_last_rank": True, "is_first_rank": True}
    )
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = Intermediate
    config = types.ModuleType("vllm.config")
    config.set_current_vllm_config = lambda value: contextlib.nullcontext()
    for name, module in {
        "vllm.config": config,
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
                scheduler.scheduled_new_reqs[0].prompt_embeds
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

    class Batch:
        def __init__(self):
            self.req_id_to_index = {}
            self.req_prompt_embeds = {}
            self.is_token_ids = torch.ones(1, 3, dtype=torch.bool)

        def remove_request(self, request_id):
            removed.append(request_id)

    class Runner:
        def __init__(self):
            self.input_batch = Batch()
            self.requests = {}

        def execute_model(self, scheduler, intermediate):
            calls.append((scheduler, intermediate))
            if scheduler.scheduled_new_reqs:
                request = scheduler.scheduled_new_reqs[0]
                self.requests[request.req_id] = types.SimpleNamespace(
                    prompt_embeds=request.prompt_embeds
                )
                self.input_batch.req_id_to_index[request.req_id] = 0
                self.input_batch.req_prompt_embeds[0] = request.prompt_embeds
            else:
                assert self.input_batch.req_prompt_embeds[0].shape == (3, 4)
            return Intermediate({"hidden_states": torch.ones(1, 4), "residual": torch.ones(1, 4)})

    worker = type("Worker", (), {"shutdown": lambda self: removed.append("shutdown")})()
    stage = VllmStage(Runner(), worker)
    first = stage.execute(torch.ones(2, 4), "request")
    second = stage.execute(first, "request")
    assert stage._positions["request"] == 3
    assert calls[1][0].scheduled_new_reqs[0].num_computed_tokens == 2
    assert calls[1][0].scheduled_new_reqs[0].prompt_embeds.shape == (3, 4)
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


def test_vllm_downstream_stage_allocates_intermediate_buffers() -> None:
    from xurcroh.adapters.vllm_v1_stage import _initialize_stage_intermediates

    buffers = Intermediate({"hidden_states": torch.empty(4, 8), "residual": torch.empty(4, 8)})
    calls = []

    class Model:
        def make_empty_intermediate_tensors(self, **kwargs):
            calls.append(kwargs)
            return buffers

    runner = types.SimpleNamespace(
        model=Model(),
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        device=torch.device("cpu"),
        max_num_tokens=4,
        intermediate_tensors=None,
    )
    _initialize_stage_intermediates(runner, 1)
    assert runner.intermediate_tensors is buffers
    assert calls == [{"batch_size": 4, "dtype": torch.bfloat16, "device": torch.device("cpu")}]


def test_vllm_cache_initialization_uses_scheduler_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from xurcroh.adapters.vllm_v1_stage import _initialize_vllm_cache

    calls = []

    def get_configs(config, specs, available):
        calls.append(("configure", specs, available))
        return ["cache-config"]

    module = types.ModuleType("vllm.v1.core.kv_cache_utils")
    module.get_kv_cache_configs = get_configs
    module.generate_scheduler_kv_cache_config = lambda configs: (
        calls.append(("schedule", configs)) or types.SimpleNamespace(num_blocks=512)
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", module)

    class Worker:
        def determine_available_memory(self):
            return 4096

        def get_kv_cache_spec(self):
            return "spec"

        def initialize_cache(self, gpu, cpu):
            calls.append(("initialize_cache", gpu, cpu))

        def initialize_from_config(self, config):
            calls.append(("initialize_from_config", config))

        def compile_or_warm_up_model(self):
            calls.append(("warmup",))

    _initialize_vllm_cache(Worker(), "config")
    assert calls == [
        ("configure", ["spec"], [4096]),
        ("schedule", ["cache-config"]),
        ("initialize_cache", 512, 0),
        ("initialize_from_config", "cache-config"),
        ("warmup",),
    ]


def test_vllm_blocks_are_allocated_scrubbed_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Batch:
        def __init__(self):
            self.req_id_to_index = {}

    runner = VllmStageRunner(type("Runner", (), {"input_batch": Batch()})())
    blocks, new = runner._request_blocks("first", 0, 17)
    assert blocks == [0, 1]
    assert new == [0, 1]
    _, new = runner._request_blocks("first", 17, 1)
    assert new == []
    _, new = runner._request_blocks("first", 32, 1)
    assert new == [2]
    cache = torch.ones(3, 2, 2)
    stream = type("Stream", (), {"synchronize": lambda self: None})()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    runner.model_runner.kv_caches = [cache]
    runner.drop_request("first")
    assert not cache.any()
    assert runner._free_blocks == set(range(256))
    reused, _ = runner._request_blocks("second", 0, 1)
    assert reused == [0]
    with pytest.raises(ProtocolError, match="already active"):
        runner._request_blocks("second", 0, 1)


def test_vllm_stage_constructs_explicit_worker_configs(
    tiny_llama_path, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shards.identities import StageIdentities
    from xurcroh.adapters.vllm_v1_stage import create_vllm_stage
    from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler

    identities = StageIdentities.create(1)
    compiled = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=469,
            compute_dtype=torch.float32,
            shard_dtype=torch.bfloat16,
        )
    ).compile()

    class Config:
        set_current_vllm_config = staticmethod(lambda value: contextlib.nullcontext())

        def __init__(self, **kwargs):
            self.max_model_len = 64
            self.is_encoder_decoder = False
            self.__dict__.update(kwargs)

    config_module = types.ModuleType("vllm.config")
    config_module.set_current_vllm_config = Config.set_current_vllm_config
    for name in (
        "CacheConfig",
        "LoadConfig",
        "ModelConfig",
        "ParallelConfig",
        "SchedulerConfig",
        "VllmConfig",
    ):
        setattr(config_module, name, Config)
    worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    created = []

    class Worker:
        def __init__(self, config, device, rank, method):
            self.config = config
            self.model_runner = types.SimpleNamespace(
                model=types.SimpleNamespace(model=types.SimpleNamespace())
            )
            created.append(self)

        def init_device(self):
            return None

        def load_model(self):
            return None

        def shutdown(self):
            return None

    worker_module.Worker = Worker
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", worker_module)
    monkeypatch.setattr("xurcroh.adapters.vllm_v1_stage.require_vllm_v1", lambda: None)
    monkeypatch.setattr("xurcroh.adapters.vllm_loader.ensure_xurcroh_loader", lambda: None)
    monkeypatch.setattr(
        "xurcroh.adapters.vllm_v1_stage._initialize_vllm_cache", lambda worker, config: None
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    stage = create_vllm_stage(
        compiled.manifest_path,
        0,
        identities.private_keys[0],
        checkpoint=tiny_llama_path,
    )
    config = created[0].config
    assert config.model_config.dtype == "bfloat16"
    assert config.model_config.enable_prompt_embeds is True
    assert config.model_config.enforce_eager is True
    assert config.cache_config.enable_prefix_caching is False
    assert config.parallel_config.pipeline_parallel_size == 1
    assert config.parallel_config.nnodes == 1
    assert config.scheduler_config.async_scheduling is False
    assert config.scheduler_config.max_model_len == 64
    assert config.scheduler_config.is_encoder_decoder is False
    assert config.load_config.load_format == "xurcroh_sharded"
    assert "stage_private_key" not in config.load_config.model_loader_extra_config
    assert config.load_config.model_loader_extra_config["stage_index"] == 0
    assert created[0].model_runner.model.model.start_layer == 0
    assert created[0].model_runner.model.model.end_layer == 2
    stage.close()


def test_hidden_from_intermediate_restores_residual_sum() -> None:
    from xurcroh.adapters.vllm_v1_stage import hidden_from_intermediate

    result = hidden_from_intermediate(
        Intermediate(
            {
                "hidden_states": torch.tensor([1.0, 2.0]),
                "residual": torch.tensor([3.0, 4.0]),
            }
        )
    )
    torch.testing.assert_close(result, torch.tensor([4.0, 6.0]))


def test_vllm_decode_reactivates_evicted_cached_request() -> None:
    from xurcroh.adapters.vllm_v1_stage import VllmStageRunner

    cached = types.SimpleNamespace(prompt_embeds=torch.ones(2, 4))
    model = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_id_to_index={}, req_prompt_embeds={}),
        requests={"request": cached},
    )
    runner = VllmStageRunner(model)
    runner._append_decode_embedding("request", torch.full((1, 4), 2.0), 2)
    assert cached.prompt_embeds.shape == (3, 4)
    assert cached.prompt_embeds[2].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_vllm_prefill_clones_client_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    from xurcroh.adapters.vllm_v1_stage import _new_request

    install_vllm_stubs(monkeypatch)
    hidden = torch.ones(2, 4)
    request = _new_request("request", hidden, [0])
    hidden.zero_()
    assert request.prompt_embeds.tolist() == [[1.0] * 4, [1.0] * 4]


def test_vllm_request_drop_refreshes_persistent_batch_metadata() -> None:
    events = []

    class Batch:
        def __init__(self):
            self.req_id_to_index = {"request": 0}
            self.req_prompt_embeds = {}

        def remove_request(self, request_id):
            events.append("remove")

        def condense(self):
            events.append("condense")

        def refresh_metadata(self):
            events.append("refresh")

    model = types.SimpleNamespace(input_batch=Batch(), requests={})
    runner = VllmStageRunner(model)
    runner.drop_request("request")
    assert events == ["remove", "condense", "refresh"]


def test_stage_worker_scopes_state_and_authenticates_cancellation(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from xurcroh.kem import HybridKemKey
    from xurcroh.runtime.stage_sessions import ClientStageSession, StageSessionStore
    from xurcroh.stage_worker import create_stage_app

    install_vllm_stubs(monkeypatch)
    stage_private = HybridKemKey.generate()
    recipient_public = stage_private.public_key
    first_session = ClientStageSession(0, recipient_public)
    second_session = ClientStageSession(0, recipient_public)
    session_id = first_session.session_id
    session_token = first_session.session_token
    dropped = []

    class Stage:
        def __init__(self):
            self.runner = types.SimpleNamespace(cache_entries=0)
            self.sessions = StageSessionStore(stage_private, 0)
            self.closed = False
            self.state_keys = []

        def execute(self, hidden, request_id, *, state_key=None):
            self.state_keys.append(state_key)
            return Intermediate({"hidden_states": hidden})

        def drop_request(self, request_id):
            dropped.append(request_id)

        def close(self):
            self.closed = True

    stage = Stage()
    monkeypatch.setattr(
        "xurcroh.adapters.vllm_v1_stage.create_vllm_stage", lambda *args, **kwargs: stage
    )
    monkeypatch.setattr(
        "xurcroh.adapters.vllm_v1_stage.decode_intermediate",
        lambda packet, receiver: {"hidden_states": torch.ones(1, 4)},
    )
    monkeypatch.setattr(
        "xurcroh.adapters.vllm_v1_stage.encode_intermediate",
        lambda tensors, request_id, sender: (b"packet", [1]),
    )
    app = create_stage_app(
        tmp_path / "manifest.json",
        stage_index=0,
        private_key=stage_private,
        engine="vllm",
        checkpoint=tmp_path,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/stages/execute",
            json={
                "request_id": "request",
                "session_id": session_id,
                "session_token": session_token,
                "client_public": first_session.client_public,
                "handshake": base64.b64encode(first_session.handshake).decode(),
                "stage_index": 0,
                "dtype": "u8",
                "shape": [1],
                "packet": base64.b64encode(b"packet").decode(),
            },
        )
        assert response.status_code == 200
        second_session_id = second_session.session_id
        second_token = second_session.session_token
        second_response = client.post(
            "/v1/stages/execute",
            json={
                "request_id": "request",
                "session_id": second_session_id,
                "session_token": second_token,
                "client_public": second_session.client_public,
                "handshake": base64.b64encode(second_session.handshake).decode(),
                "stage_index": 0,
                "dtype": "u8",
                "shape": [1],
                "packet": base64.b64encode(b"packet").decode(),
            },
        )
        assert second_response.status_code == 200
        assert stage.state_keys == [f"{session_id}:request", f"{second_session_id}:request"]
        assert client.delete("/v1/stages/request").status_code == 403
        headers = {
            "X-Xurcroh-Session-Id": session_id,
            "X-Xurcroh-Session-Token": session_token,
        }
        assert client.delete("/v1/stages/request", headers=headers).status_code == 204
        assert dropped == [f"{session_id}:request"]
        assert client.delete(f"/v1/stages/sessions/{session_id}").status_code == 403
        assert (
            client.delete(f"/v1/stages/sessions/{session_id}", headers=headers).status_code == 204
        )
        second_headers = {
            "X-Xurcroh-Session-Id": second_session_id,
            "X-Xurcroh-Session-Token": second_token,
        }
        assert (
            client.delete(
                f"/v1/stages/sessions/{second_session_id}", headers=second_headers
            ).status_code
            == 204
        )
        assert stage.sessions.active_count == 0


def test_stage_transition_also_transforms_residual_basis() -> None:
    import threading

    from xurcroh.config import WorkerConfig
    from xurcroh.requests import CompletionRequest
    from xurcroh.runtime.stage import DenseStageEngine

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[2, 3]])}

        def decode(self, token_ids, skip_special_tokens):
            return "x"

    class Boundary:
        def embed(self, token_ids):
            return torch.ones(1, 1, 4)

        def transition(self, hidden, stage_index):
            assert stage_index == 0
            return hidden * 2

        def sample(self, hidden, request):
            assert hidden.reshape(-1).tolist() == [5, 5, 5, 5]
            return torch.tensor([1])

    engine = DenseStageEngine.__new__(DenseStageEngine)
    engine._closed = False
    engine.workers = [
        WorkerConfig(name="a", endpoint="https://a.invalid"),
        WorkerConfig(name="b", endpoint="https://b.invalid"),
    ]
    engine._tokenizer = Tokenizer()
    engine._stop_token_ids = {Tokenizer.eos_token_id}
    engine.position_limit = 100
    engine.vocabulary_size = 100
    engine.boundary = Boundary()
    engine._residuals = {}

    def execute(index, hidden, request_id, session, timeout, circuit_id=None):
        engine._residuals.setdefault(request_id, {})[index] = torch.full_like(hidden, 3)
        return hidden

    engine._execute_stage = execute
    request = CompletionRequest.validated(
        model="tiny",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        request_id="residual-transition",
    )
    list(
        engine._generate(
            request,
            cancelled=threading.Event(),
            timeout=60,
            circuit_id=None,
            request_id="residual-transition",
            sessions=[None, None],
        )
    )
    assert engine._residuals["residual-transition"][0].reshape(-1).tolist() == [6, 6, 6, 6]


def test_vllm_decode_uses_prompt_embedding_not_placeholder_token() -> None:
    from xurcroh.adapters.vllm_v1_stage import VllmStageRunner

    batch = types.SimpleNamespace(
        req_id_to_index={"request": 0},
        req_prompt_embeds={0: torch.ones(9, 4)},
        is_token_ids=torch.ones(1, 10, dtype=torch.bool),
    )
    cached = types.SimpleNamespace(prompt_embeds=torch.ones(9, 4))
    model = types.SimpleNamespace(input_batch=batch, requests={"request": cached})
    runner = VllmStageRunner(model)
    runner._append_decode_embedding("request", torch.full((1, 4), 2.0), 9)
    assert not batch.is_token_ids[0, 9]
    assert cached.prompt_embeds.shape == (10, 4)


def test_vllm_handoff_preserves_unfused_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    from xurcroh.config import WorkerConfig
    from xurcroh.runtime.stage import DenseStageEngine

    engine = DenseStageEngine.__new__(DenseStageEngine)
    engine.workers = [WorkerConfig(name="a", endpoint="https://a.invalid")]
    engine._residuals = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"stage_index": 0, "dtype": "u8", "packet": "cGFja2V0"}

    engine._client = types.SimpleNamespace(post=lambda *args, **kwargs: Response())
    session = types.SimpleNamespace(
        session_id="session",
        session_token="token",
        client_public="client",
        handshake=bytes(48),
        sender=object(),
        receiver=object(),
    )
    monkeypatch.setattr("xurcroh.runtime.stage.encode_intermediate", lambda *args: (b"packet", [1]))
    monkeypatch.setattr(
        "xurcroh.runtime.stage.decode_intermediate",
        lambda *args: {
            "hidden_states": torch.tensor([2.0]),
            "residual": torch.tensor([3.0]),
        },
    )

    hidden = engine._execute_vllm_stage(0, torch.tensor([1.0]), "request", session, 60, None)

    assert hidden.tolist() == [2.0]
    assert engine._residuals["request"][0].tolist() == [3.0]
