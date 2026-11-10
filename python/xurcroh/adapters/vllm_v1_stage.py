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


def _new_request(request_id: str, hidden: torch.Tensor) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=request_id,
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=((),),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=hidden.detach().unsqueeze(0),
    )


def _scheduler(request_id: str, hidden: torch.Tensor, position: int) -> Any:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    if position == 0:
        cached = CachedRequestData.make_empty()
        new = [_new_request(request_id, hidden)]
    else:
        cached = CachedRequestData(
            req_ids=[request_id],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={request_id: []},
            new_block_ids=[None],
            num_computed_tokens=[position],
            num_output_tokens=[0],
        )
        new = []
    tokens = hidden.shape[0] if position == 0 else 1
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

    def execute(
        self,
        hidden: torch.Tensor,
        request_id: str,
        position: int,
        intermediate: Any = None,
    ) -> Any:
        from vllm.distributed.parallel_state import GroupCoordinator
        from vllm.sequence import IntermediateTensors

        require_vllm_v1()
        if position < 0 or (position > 0 and hidden.shape[0] != 1):
            raise ProtocolError("vLLM decode must contain exactly one token")
        scheduler = _scheduler(request_id, hidden, position)
        with (
            _EXECUTE_LOCK,
            patch.object(
                GroupCoordinator, "is_last_rank", new_callable=PropertyMock(return_value=False)
            ),
        ):
            output = self.model_runner.execute_model(scheduler, intermediate)
        if not isinstance(output, IntermediateTensors):
            raise ProtocolError("vLLM stage did not return IntermediateTensors")
        return output

    def drop_request(self, request_id: str) -> None:
        remove = getattr(self.model_runner.input_batch, "remove_request", None)
        if callable(remove):
            remove(request_id)


class VllmStage:
    def __init__(self, model_runner: Any, worker: Any = None) -> None:
        self.runner = VllmStageRunner(model_runner)
        self.worker = worker
        self.sessions: Any = None
        self.stage_index = 0
        self.closed = False

    def execute(self, hidden: Any, request_id: str | None = None) -> Any:
        if self.closed:
            raise ModelLoadError("vLLM stage is closed")
        if request_id is None:
            raise ProtocolError("vLLM stage execution requires a request ID")
        position = getattr(self, "_positions", {}).get(request_id, 0)
        intermediate = hidden if hasattr(hidden, "tensors") else None
        tensor = hidden.tensors["hidden_states"] if intermediate is not None else hidden
        output = self.runner.execute(tensor, request_id, position, intermediate)
        if not hasattr(self, "_positions"):
            self._positions = {}
        self._positions[request_id] = position + tensor.shape[0]
        return output

    def drop_request(self, request_id: str) -> None:
        getattr(self, "_positions", {}).pop(request_id, None)
        self.runner.drop_request(request_id)

    def close(self) -> None:
        if not self.closed:
            if hasattr(self, "_positions"):
                self._positions.clear()
            if hasattr(self, "sessions"):
                self.sessions.close()
            if self.worker is not None:
                self.worker.shutdown()
            self.closed = True


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
    os.environ.setdefault("RANK", str(stage_index))
    os.environ.setdefault("WORLD_SIZE", str(stage_count))
    os.environ.setdefault("LOCAL_RANK", str(device))
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_worker import Worker

    from xurcroh.runtime.stage_sessions import StageSessionStore

    identity = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    config = VllmConfig(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        load_format="xurcroh_sharded",
        model_loader_extra_config={
            "manifest_path": str(path),
            "stage_index": stage_index,
            "stage_private_key": identity,
        },
        pipeline_parallel_size=stage_count,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        enable_prompt_embeds=True,
    )
    worker = Worker(config, device, stage_index, "env://")
    try:
        worker.init_device()
        worker.load_model()
        worker.initialize_cache(256, 0)
        worker.initialize_from_config(worker.get_kv_cache_spec())
        stage = VllmStage(worker.model_runner, worker)
        stage.stage_index = stage_index
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
    return cast(torch.Tensor, tensors["hidden_states"])


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
