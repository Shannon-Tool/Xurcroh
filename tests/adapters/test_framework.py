from __future__ import annotations

import pytest

from xurcroh.adapters.framework import FrameworkAdapter, ParallelismConfig
from xurcroh.errors import CompatibilityError


def test_vllm_and_sglang_commands_enforce_security_flags() -> None:
    vllm = FrameworkAdapter("vllm")
    command = vllm.worker_command(
        model="/models/llama",
        pipeline_parallel=2,
        tensor_parallel=1,
    )
    assert "--load-format xurcroh_sharded" in command
    assert "--no-enable-prefix-caching" in command
    assert "--no-async-scheduling" in command

    sglang = FrameworkAdapter("sglang")
    command = sglang.worker_command(
        model="/models/deepseek",
        pipeline_parallel=4,
        tensor_parallel=2,
    )
    assert "--disable-radix-cache" in command
    assert "--chunked-prefill-size 4096" in command
    assert "--enable-dynamic-chunking" in command


def test_xdit_uses_official_runner_and_validates_parallelism() -> None:
    adapter = FrameworkAdapter("xdit")
    command = adapter.worker_command(
        model="FLUX.1-dev",
        parallelism=ParallelismConfig(
            ring_degree=0,
            ulysses_degree=4,
            attention_backend="flash",
        ),
    )
    assert command.startswith("xdit --model FLUX.1-dev")
    assert "--ulysses_degree 4" in command
    with pytest.raises(CompatibilityError):
        FrameworkAdapter("missing")


def test_missing_framework_fails_closed() -> None:
    with pytest.raises(CompatibilityError, match="requires"):
        FrameworkAdapter("vllm").require_installed()
