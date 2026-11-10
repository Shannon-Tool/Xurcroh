from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shards.identities import StageIdentities
from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError
from xurcroh.runtime.tensor_codec import decode_stage_tensor, encode_stage_tensor


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_native_dtype_stage_round_trip(dtype: torch.dtype) -> None:
    key = bytes(range(32))
    sender = PipelineTensorInterceptor(key, b"client", b"stage", b"dtype-test")
    receiver = PipelineTensorInterceptor(key, b"stage", b"client", b"dtype-test")
    hidden = torch.randn(3, 5, dtype=torch.float32).to(dtype)
    packet, dtype_name, shape = encode_stage_tensor(hidden, "request", sender)
    decoded = decode_stage_tensor(packet, receiver, expected_dtype=dtype_name, expected_shape=shape)
    torch.testing.assert_close(decoded, hidden)


def test_native_dtype_rejects_wrong_shape_and_truncated_payload() -> None:
    key = bytes(range(32))
    sender = PipelineTensorInterceptor(key, b"client", b"stage", b"dtype-test")
    receiver = PipelineTensorInterceptor(key, b"stage", b"client", b"dtype-test")
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    packet, _, _ = encode_stage_tensor(hidden, "request", sender)
    with pytest.raises(ProtocolError, match="descriptor"):
        decode_stage_tensor(packet, receiver, expected_shape=[4, 2])


def test_compiler_emits_bf16_shards_after_fp64_transformation(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    import json

    from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler
    from xurcroh.shards.format import read_shard
    from xurcroh.shards.recipients import open_stage_key

    identities = StageIdentities.create(1)
    result = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=431,
            compute_dtype=torch.float64,
            shard_dtype=torch.bfloat16,
        )
    ).compile()
    manifest = json.loads(result.manifest_path.read_text())
    envelope = json.loads(result.key_envelopes[0].read_text())
    key = open_stage_key(
        envelope,
        identities.private_keys[0],
        manifest_digest=manifest["digest"],
        checkpoint_fingerprint=manifest["checkpoint_fingerprint"],
    )
    tensors, _ = read_shard(result.stage_shards[0], key)
    assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    boundary, _ = read_shard(result.boundary_shard, identities.boundary_key)
    assert boundary["basis.input"].dtype == torch.bfloat16


def test_public_sdk_executes_bf16_remote_stage(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from integration.test_stage_sdk import running_stage
    from Xurcroh import Xurcroh
    from xurcroh.shards.compiler import CompilationConfig, XurcrohWeightCompiler

    identities = StageIdentities.create(1)
    compiled = XurcrohWeightCompiler(
        CompilationConfig(
            checkpoint=tiny_llama_path,
            output_dir=tmp_path / "compiled",
            stage_count=1,
            boundary_key=identities.boundary_key,
            recipients=identities.recipients,
            seed=433,
            compute_dtype=torch.float64,
            shard_dtype=torch.bfloat16,
        )
    ).compile()
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    with (
        running_stage(compiled.manifest_path, 0, identities.private_keys[0]) as endpoint,
        Xurcroh(
            cluster={"name": "stage-0", "endpoint": endpoint},
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
            security={"mode": "development"},
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0,
        )
    assert response.usage.completion_tokens == 1
