from __future__ import annotations

import pytest

from Xurcroh import Xurcroh
from xurcroh.errors import ClusterUnavailableError


def test_remote_worker_performs_actual_sync_and_streaming_inference(running_worker) -> None:
    with Xurcroh(
        cluster={"endpoint": running_worker, "security": {"mode": "development"}},
        model_path={"tiny": "remote-tiny"},
    ) as client:
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=3,
            temperature=0,
        )
        assert response.choices[0].message.content is not None
        assert response.usage.completion_tokens > 0
        assert response.execution.engine == "remote-openai"
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=3,
            temperature=0,
            stream=True,
        )
        chunks = list(stream)
        assert any(chunk.choices[0].delta.content for chunk in chunks)
        assert chunks[-1].choices[0].finish_reason is not None


def test_remote_failure_uses_alternate_worker(running_worker) -> None:
    with Xurcroh(
        cluster=[
            {"name": "dead", "endpoint": "http+dev://127.0.0.1:1"},
            {"name": "live", "endpoint": running_worker},
        ],
        model_path={"tiny": "remote-tiny"},
        security={"mode": "development"},
    ) as client:
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
        assert response.usage.completion_tokens == 1


def test_strict_mode_rejects_development_http(running_worker) -> None:
    with pytest.raises(ClusterUnavailableError, match="strict"):
        Xurcroh(cluster={"endpoint": running_worker}, model_path={"tiny": "remote-tiny"})


def test_strict_mode_rejects_full_remote_checkpoint_without_stage_manifest() -> None:
    with pytest.raises(ClusterUnavailableError, match="stage manifest"):
        Xurcroh(
            cluster={"endpoint": "https://worker.example:9443"},
            model_path={"tiny": "remote-tiny"},
        )
