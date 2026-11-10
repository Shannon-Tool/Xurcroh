from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.errors import RequestTimeoutError


def test_admission_backpressure_and_privacy_safe_metrics(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu", "resources": {"max_concurrent_requests": 1}},
        model_path={"tiny": tiny_model_path},
    ) as client:
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=4,
            stream=True,
        )
        next(stream)
        with pytest.raises(RequestTimeoutError, match="admission"):
            client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
                timeout=1e-9,
            )
        stream.close()
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
        assert response.usage.completion_tokens == 1
        metrics = client._runtime.metrics()
        assert metrics["requests_completed"] == 1
        assert metrics["requests_timed_out"] == 1
        assert "hello" not in str(metrics)
