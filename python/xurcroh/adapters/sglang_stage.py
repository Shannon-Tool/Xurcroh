from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock, patch

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from xurcroh.adapters.sglang_kv import scrub_sglang_request
from xurcroh.errors import CompatibilityError, ModelLoadError, ProtocolError
from xurcroh.shards.recipients import verify_manifest_digest

SGLANG_VERSION = "0.5.23"
_EXECUTE_LOCK = threading.Lock()


def require_sglang() -> None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version("sglang")
    except PackageNotFoundError as error:
        raise CompatibilityError("SGLang stage execution requires sglang==0.5.23") from error
    if installed != SGLANG_VERSION:
        raise CompatibilityError(
            f"unsupported SGLang version {installed}; Xurcroh requires {SGLANG_VERSION}"
        )


def _forward_batch(request_id: str, hidden: torch.Tensor, position: int) -> Any:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    tokens = hidden.shape[0]
    device = hidden.device
    total = position + tokens
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND if position == 0 else ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.zeros(tokens, dtype=torch.long, device=device),
        req_pool_indices=torch.zeros(1, dtype=torch.long, device=device),
        seq_lens=torch.tensor([total], dtype=torch.long, device=device),
        out_cache_loc=torch.arange(position, total, dtype=torch.long, device=device),
        seq_lens_sum=total,
        input_embeds=hidden,
    )


class SGLangStageRunner:
    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def execute(self, hidden: torch.Tensor, request_id: str, position: int, proxy: Any) -> Any:
        require_sglang()
        if position < 0 or (position > 0 and hidden.shape[0] != 1):
            raise ProtocolError("SGLang decode must contain exactly one token")
        batch = _forward_batch(request_id, hidden, position)
        with (
            _EXECUTE_LOCK,
            patch.object(
                type(self.worker.pp_group),
                "is_last_rank",
                new_callable=PropertyMock(return_value=False),
            ),
        ):
            result = self.worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)
        proxy_output = getattr(result, "pp_hidden_states_proxy_tensors", None)
        if proxy_output is None:
            raise ProtocolError("SGLang stage did not return PPProxyTensors")
        return proxy_output

    def drop_request(self, request_id: str, position: int = 0) -> None:
        scrub_sglang_request(self.worker, 0, range(position))


class SGLangStage:
    def __init__(self, worker: Any, stage_index: int) -> None:
        self.worker = worker
        self.runner = SGLangStageRunner(worker)
        self.sessions: Any = None
        self.stage_index = stage_index
        self.closed = False
        self._positions: dict[str, int] = {}

    def execute(
        self, hidden: Any, request_id: str | None = None, *, state_key: str | None = None
    ) -> Any:
        if self.closed:
            raise ModelLoadError("SGLang stage is closed")
        if request_id is None:
            raise ProtocolError("SGLang stage execution requires a request ID")
        state_key = state_key or request_id
        position = self._positions.get(state_key, 0)
        proxy = hidden if hasattr(hidden, "tensors") else None
        tensor = hidden.tensors["hidden_states"] if proxy is not None else hidden
        output = self.runner.execute(tensor, state_key, position, proxy)
        self._positions[state_key] = position + tensor.shape[0]
        return output

    def drop_request(self, request_id: str) -> None:
        position = self._positions.pop(request_id, 0)
        self.runner.drop_request(request_id, position)

    def close(self) -> None:
        if not self.closed:
            for request_id in list(self._positions):
                self.drop_request(request_id)
            if self.sessions is not None:
                self.sessions.close()
            self.closed = True


def create_sglang_stage(
    manifest_path: str | Path,
    stage_index: int,
    private_key: X25519PrivateKey,
    *,
    checkpoint: str | Path,
    device: int = 0,
) -> SGLangStage:
    require_sglang()
    if not torch.cuda.is_available():
        raise CompatibilityError("SGLang stage execution requires a CUDA device")
    path = Path(manifest_path)
    document = json.loads(path.read_text())
    verify_manifest_digest(document)
    if document["format"] != "xurcroh-manifest" or document["version"] != 2:
        raise ModelLoadError("SGLang requires a dense manifest v2")
    stage_count = int(document["stage_count"])
    if stage_index < 0 or stage_index >= stage_count:
        raise ModelLoadError("SGLang stage index is out of range")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29616")
    os.environ.setdefault("RANK", str(stage_index))
    os.environ.setdefault("WORLD_SIZE", str(stage_count))
    os.environ.setdefault("LOCAL_RANK", str(device))
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs

    from xurcroh.runtime.stage_sessions import StageSessionStore

    identity = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    server_args = ServerArgs(
        model_path=str(checkpoint),
        tokenizer_path=str(checkpoint),
        pipeline_parallel_size=stage_count,
        tensor_parallel_size=1,
        disable_radix_cache=True,
        chunked_prefill_size=4096,
        enable_dynamic_chunking=True,
        load_format="private",
        model_loader_extra_config={
            "manifest_path": str(path),
            "stage_index": stage_index,
            "stage_private_key": identity,
        },
        enforce_eager=True,
    )
    worker = TpModelWorker(server_args, device, 29616)
    worker.alloc_memory_pool()
    worker.init_attention_backends()
    worker.init_cuda_graphs(capture_decode_cuda_graph=False)
    stage = SGLangStage(worker, stage_index)
    stage.sessions = StageSessionStore(private_key, stage_index)
    return stage


def tensors_from_proxy(proxy: Any) -> dict[str, torch.Tensor]:
    tensors = getattr(proxy, "tensors", None)
    if not isinstance(tensors, dict) or "hidden_states" not in tensors:
        raise ProtocolError("SGLang proxy does not contain hidden_states")
    result = {"hidden_states": tensors["hidden_states"]}
    residual = tensors.get("residual")
    if isinstance(residual, torch.Tensor):
        result["residual"] = residual
    return result
