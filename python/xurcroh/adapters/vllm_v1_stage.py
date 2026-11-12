from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, cast
from unittest.mock import PropertyMock, patch

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from xurcroh.errors import CompatibilityError, ModelLoadError, ProtocolError
from xurcroh.shards.recipients import verify_manifest_digest

VLLM_VERSION = "0.15.0"
_EXECUTE_LOCK = threading.Lock()


def require_vllm_v1() -> None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version("vllm")
    except PackageNotFoundError as error:
        raise CompatibilityError("vLLM stage execution requires vllm==0.15.0") from error
    if installed != VLLM_VERSION:
        raise CompatibilityError(
            f"unsupported vLLM version {installed}; Xurcroh requires {VLLM_VERSION}"
        )


def _new_request(
    request_id: str, hidden: torch.Tensor, block_ids: list[int], num_computed_tokens: int = 0
) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=request_id,
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=(block_ids,),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prompt_embeds=hidden.detach().clone(),
    )


def _scheduler(
    request_id: str,
    hidden: torch.Tensor,
    position: int,
    block_ids: list[int],
    new_blocks: list[int],
) -> Any:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    cached = CachedRequestData.make_empty()
    new = [_new_request(request_id, hidden, block_ids, position)]
    tokens = 1 if position > 0 else hidden.shape[0]
    return SchedulerOutput(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={request_id: tokens},
        total_num_scheduled_tokens=tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class VllmStageRunner:
    def __init__(self, model_runner: Any) -> None:
        self.model_runner = model_runner
        cache_config = getattr(model_runner, "cache_config", None)
        self.block_size = int(getattr(cache_config, "block_size", 16) or 16)
        block_count = int(getattr(cache_config, "num_gpu_blocks", 256) or 256)
        self._free_blocks = set(range(block_count))
        self._blocks: dict[str, list[int]] = {}
        self.stage_index = 0

    def _request_blocks(
        self, request_id: str, position: int, tokens: int
    ) -> tuple[list[int], list[int]]:
        if position == 0 and request_id in self._blocks:
            raise ProtocolError("vLLM request ID is already active")
        blocks = self._blocks.setdefault(request_id, [])
        required = (position + tokens + self.block_size - 1) // self.block_size
        new_blocks = []
        while len(blocks) < required:
            if not self._free_blocks:
                raise ProtocolError("vLLM stage has no free KV blocks")
            block = min(self._free_blocks)
            self._free_blocks.remove(block)
            blocks.append(block)
            new_blocks.append(block)
        return blocks, new_blocks

    def _append_decode_embedding(
        self, request_id: str, hidden: torch.Tensor, position: int
    ) -> None:
        batch = self.model_runner.input_batch
        index = batch.req_id_to_index.get(request_id)
        request = self.model_runner.requests.get(request_id)
        if request is None:
            active = list(batch.req_id_to_index)
            raise ProtocolError(f"vLLM decode request {request_id!r} has no cached state: {active}")
        existing = (
            batch.req_prompt_embeds.get(index) if index is not None else request.prompt_embeds
        )
        if existing is None or existing.shape != (position, hidden.shape[1]):
            raise ProtocolError("vLLM decode embedding state is inconsistent")
        updated = torch.cat([existing, hidden.detach()], dim=0)
        request.prompt_embeds = updated
        if index is not None:
            batch.req_prompt_embeds[index] = updated
            batch.is_token_ids[index, position] = False

    def execute(
        self,
        hidden: torch.Tensor,
        request_id: str,
        position: int,
        intermediate: Any = None,
    ) -> Any:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import GroupCoordinator
        from vllm.sequence import IntermediateTensors

        require_vllm_v1()
        if position < 0 or (position > 0 and hidden.shape[0] != 1):
            raise ProtocolError("vLLM decode must contain exactly one token")
        blocks, new_blocks = self._request_blocks(request_id, position, hidden.shape[0])
        if position > 0:
            self._append_decode_embedding(request_id, hidden, position)
            hidden = self.model_runner.requests[request_id].prompt_embeds
        scheduler = _scheduler(request_id, hidden, position, blocks, new_blocks)
        with (
            _EXECUTE_LOCK,
            set_current_vllm_config(getattr(self.model_runner, "vllm_config", None)),
            patch.object(
                GroupCoordinator, "is_last_rank", new_callable=PropertyMock(return_value=False)
            ),
            patch.object(
                GroupCoordinator,
                "is_first_rank",
                new_callable=PropertyMock(return_value=self.stage_index == 0),
            ),
        ):
            output = self.model_runner.execute_model(scheduler, intermediate)
        if not isinstance(output, IntermediateTensors):
            raise ProtocolError("vLLM stage did not return IntermediateTensors")
        return output

    def drop_request(self, request_id: str) -> None:
        batch = getattr(self.model_runner, "input_batch", None)
        index = getattr(batch, "req_id_to_index", {}).get(request_id)
        with torch.inference_mode():
            if batch is not None and index is not None:
                embedding = batch.req_prompt_embeds.pop(index, None)
                if embedding is not None:
                    embedding.zero_()
            request = getattr(self.model_runner, "requests", {}).pop(request_id, None)
            prompt_embeds = getattr(request, "prompt_embeds", None)
            if prompt_embeds is not None:
                prompt_embeds.zero_()
            blocks = self._blocks.pop(request_id, [])
            for cache in getattr(self.model_runner, "kv_caches", []):
                for block in blocks:
                    cache[block].zero_()
        if blocks and getattr(self.model_runner, "kv_caches", []):
            torch.cuda.current_stream().synchronize()
        self._free_blocks.update(blocks)
        remove = getattr(batch, "remove_request", None)
        if callable(remove):
            remove(request_id)
        condense = getattr(batch, "condense", None)
        if callable(condense):
            condense()
        refresh = getattr(batch, "refresh_metadata", None)
        if callable(refresh):
            refresh()


class VllmStage:
    def __init__(self, model_runner: Any, worker: Any = None) -> None:
        self.runner = VllmStageRunner(model_runner)
        self.worker = worker
        self.sessions: Any = None
        self.stage_index = 0
        self.closed = False

    def execute(
        self, hidden: Any, request_id: str | None = None, *, state_key: str | None = None
    ) -> Any:
        if self.closed:
            raise ModelLoadError("vLLM stage is closed")
        if request_id is None:
            raise ProtocolError("vLLM stage execution requires a request ID")
        state_key = state_key or request_id
        position = getattr(self, "_positions", {}).get(state_key, 0)
        intermediate = hidden if hasattr(hidden, "tensors") else None
        tensor = hidden.tensors["hidden_states"] if intermediate is not None else hidden
        output = self.runner.execute(tensor, state_key, position, intermediate)
        if not hasattr(self, "_positions"):
            self._positions = {}
        self._positions[state_key] = position + tensor.shape[0]
        return output

    def drop_request(self, request_id: str) -> None:
        getattr(self, "_positions", {}).pop(request_id, None)
        self.runner.drop_request(request_id)

    def close(self) -> None:
        if not self.closed:
            for request_id in list(getattr(self, "_positions", {})):
                self.drop_request(request_id)
            if self.sessions is not None:
                self.sessions.close()
            if self.worker is not None:
                self.worker.shutdown()
            self.closed = True


def _initialize_stage_intermediates(model_runner: Any, stage_index: int) -> None:
    if stage_index == 0 or model_runner.intermediate_tensors is not None:
        return
    model_runner.intermediate_tensors = model_runner.model.make_empty_intermediate_tensors(
        batch_size=model_runner.max_num_tokens,
        dtype=model_runner.model_config.dtype,
        device=model_runner.device,
    )


def _initialize_vllm_cache(worker: Any, config: Any) -> None:
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
    )

    available = worker.determine_available_memory()
    cache_configs = get_kv_cache_configs(config, [worker.get_kv_cache_spec()], [available])
    if len(cache_configs) != 1:
        raise ModelLoadError("vLLM returned an invalid pipeline KV cache configuration")
    scheduler_config = generate_scheduler_kv_cache_config(cache_configs)
    worker.initialize_cache(scheduler_config.num_blocks, 0)
    worker.initialize_from_config(cache_configs[0])
    worker.compile_or_warm_up_model()


def _set_stage_layer_range(model: Any, layer_indices: list[int]) -> None:
    if not layer_indices:
        raise ModelLoadError("stage layer range cannot be empty")
    stage_model = getattr(model, "model", None)
    if stage_model is None:
        raise ModelLoadError("vLLM model does not expose pipeline layer ranges")
    stage_model.start_layer = min(layer_indices)
    stage_model.end_layer = max(layer_indices) + 1


def create_vllm_stage(
    manifest_path: str | Path,
    stage_index: int,
    private_key: X25519PrivateKey,
    *,
    checkpoint: str | Path,
    device: int = 0,
) -> VllmStage:
    require_vllm_v1()
    if not torch.cuda.is_available():
        raise CompatibilityError("vLLM stage execution requires a CUDA device")
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    verify_manifest_digest(document)
    if document["format"] != "xurcroh-manifest" or document["version"] != 2:
        raise ModelLoadError("vLLM requires a dense manifest v2")
    stage_count = int(document["stage_count"])
    if stage_index < 0 or stage_index >= stage_count:
        raise ModelLoadError("vLLM stage index is out of range")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29615")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ.setdefault("LOCAL_RANK", str(device))
    from vllm.config import (
        CacheConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.v1.worker.gpu_worker import Worker

    from xurcroh.adapters.vllm_loader import ensure_xurcroh_loader
    from xurcroh.runtime.stage_sessions import StageSessionStore

    ensure_xurcroh_loader()
    storage_dtype = document["stages"][stage_index].get("storage_dtype", "bf16")
    dtype = {
        "f64": "float32",
        "f32": "float32",
        "bf16": "bfloat16",
        "f16": "float16",
    }.get(storage_dtype)
    if dtype is None:
        raise CompatibilityError(f"unsupported vLLM shard dtype: {storage_dtype}")
    identity = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    model_config = ModelConfig(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        enforce_eager=True,
        enable_prompt_embeds=True,
        dtype=dtype,
    )
    config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=False),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            nnodes=1,
            node_rank=0,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
            async_scheduling=False,
        ),
        load_config=LoadConfig(
            load_format="xurcroh_sharded",
            model_loader_extra_config={
                "manifest_path": str(path),
                "stage_index": stage_index,
                "stage_private_key": identity,
            },
        ),
    )
    worker = Worker(config, device, 0, "env://")
    try:
        worker.init_device()
        worker.load_model()
        _set_stage_layer_range(
            worker.model_runner.model, document["stages"][stage_index]["layer_indices"]
        )
        _initialize_stage_intermediates(worker.model_runner, stage_index)
        with set_current_vllm_config(config):
            _initialize_vllm_cache(worker, config)
        stage = VllmStage(worker.model_runner, worker)
        stage.stage_index = stage_index
        stage.runner.stage_index = stage_index
        stage.sessions = StageSessionStore(private_key, stage_index)
        return stage
    except Exception:
        worker.shutdown()
        raise


def hidden_from_intermediate(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    tensors = getattr(value, "tensors", None)
    if not isinstance(tensors, dict) or "hidden_states" not in tensors:
        raise ProtocolError("stage output does not contain hidden_states")
    hidden = cast(torch.Tensor, tensors["hidden_states"])
    residual = tensors.get("residual")
    if isinstance(residual, torch.Tensor):
        hidden = hidden + residual
    return hidden


def encode_intermediate(
    tensors: dict[str, torch.Tensor],
    request_id: str,
    sender: Any,
) -> tuple[bytes, list[int]]:
    from safetensors.torch import save

    payload = save({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()})
    return sender.seal(request_id, "u8", [len(payload)], payload), [len(payload)]


def decode_intermediate(packet: bytes, receiver: Any) -> dict[str, torch.Tensor]:
    from safetensors.torch import load

    dtype, shape, payload = receiver.open_tensor(packet)
    if dtype != "u8" or shape != [len(payload)]:
        raise ProtocolError("vLLM intermediate bundle descriptor is invalid")
    tensors = load(payload)
    if set(tensors) - {"hidden_states", "residual"} or "hidden_states" not in tensors:
        raise ProtocolError("vLLM intermediate bundle has invalid tensor names")
    return tensors
