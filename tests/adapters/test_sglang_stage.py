from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from xurcroh.adapters.sglang_kv import scrub_sglang_request
from xurcroh.adapters.sglang_stage import (
    SGLangStageRunner,
    require_sglang,
    tensors_from_proxy,
)
from xurcroh.errors import CompatibilityError


class Proxy:
    def __init__(self, tensors):
        self.tensors = tensors


def test_sglang_forward_and_proxy_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForwardBatch:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class ForwardMode:
        EXTEND = "extend"
        DECODE = "decode"

    forward_module = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_module.ForwardBatch = ForwardBatch
    forward_module.ForwardMode = ForwardMode
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info", forward_module)
    monkeypatch.setattr("xurcroh.adapters.sglang_stage.require_sglang", lambda: None)
    calls = []

    class Worker:
        pp_group = type("Group", (), {"is_last_rank": True})()

        def forward_batch_generation(self, batch, pp_proxy_tensors=None):
            calls.append((batch, pp_proxy_tensors))
            proxy = pp_proxy_tensors or Proxy(
                {
                    "hidden_states": batch.input_embeds,
                    "residual": batch.input_embeds.clone(),
                }
            )
            return SimpleNamespace(pp_hidden_states_proxy_tensors=proxy)

    output = SGLangStageRunner(Worker()).execute(
        torch.randn(2, 4, dtype=torch.bfloat16), "request", 0, None
    )
    tensors = tensors_from_proxy(output)
    assert tensors["hidden_states"].shape == (2, 4)
    assert tensors["residual"].shape == (2, 4)
    batch, _ = calls[0]
    assert batch.seq_lens_sum == 2
    assert batch.out_cache_loc.tolist() == [0, 1]


def test_sglang_version_is_required() -> None:
    with pytest.raises(CompatibilityError, match=r"sglang==0\.5\.23"):
        require_sglang()


def test_sglang_private_loader_opens_only_recipient_shard(tiny_llama_path, tmp_path) -> None:
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    from sglang.private.private_model_loader import PrivateModelLoader
    from shards.identities import StageIdentities

    identities = StageIdentities.create(1)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=457)
    loader = object.__new__(PrivateModelLoader)
    loader.load_config = SimpleNamespace(
        model_loader_extra_config={
            "manifest_path": str(compiled.manifest_path),
            "stage_index": 0,
            "stage_private_key": identities.private_keys[0].private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()
            ),
        }
    )
    tensors = dict(loader._get_all_weights(SimpleNamespace(), SimpleNamespace()))
    assert "model.layers.0.self_attn.q_proj.weight" in tensors
    loader.download_model(SimpleNamespace())


def test_sglang_kv_cleanup_zeroes_slots_before_free() -> None:
    req_to_token = torch.ones(2, 4, dtype=torch.int32)
    k_buffer = torch.ones(1, 4, 2, dtype=torch.float32)
    v_buffer = torch.ones_like(k_buffer)
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=req_to_token,
            free_rows=lambda indices: freed.extend(indices),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            kv_cache=SimpleNamespace(k_buffer=k_buffer, v_buffer=v_buffer)
        ),
    )
    worker = SimpleNamespace(model_runner=runner)
    freed = []
    scrub_sglang_request(worker, 0, [0, 2])
    assert not req_to_token[0].any()
    assert not k_buffer[:, [0, 2]].any()
    assert not v_buffer[:, [0, 2]].any()
    assert freed == [0]
