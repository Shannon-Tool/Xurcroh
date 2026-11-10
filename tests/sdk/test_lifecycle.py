from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError, RequestCancelledError, RequestTimeoutError


def test_cancellation_uses_public_request_id(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            stream=True,
            request_id="cancel-me",
        )
        next(stream)
        assert client.cancel("cancel-me")
        with pytest.raises(RequestCancelledError):
            list(stream)
        assert not client.cancel("cancel-me")


def test_timeout_is_a_typed_error(tiny_model_path: Path) -> None:
    with (
        Xurcroh(
            cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
        ) as client,
        pytest.raises(RequestTimeoutError),
    ):
        client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2,
            timeout=1e-9,
        )


def test_duplicate_active_request_id_is_rejected(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=4,
            stream=True,
            request_id="duplicate",
        )
        next(stream)
        with pytest.raises(ConfigurationError, match="already active"):
            client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
                request_id="duplicate",
            )
        stream.close()


def test_concurrent_requests_return_independent_usage(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:

        def complete(index: int) -> tuple[str, int]:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": f"hello {index % 2}"}],
                max_tokens=2,
                temperature=0,
            )
            return response.execution.request_id, response.usage.total_tokens

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(complete, range(3)))
        assert len({request_id for request_id, _ in results}) == 3
        assert all(tokens > 0 for _, tokens in results)


def test_close_does_not_deadlock_with_abandoned_stream(tiny_model_path: Path) -> None:
    client = Xurcroh(cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path})
    stream = client.chat.completions.create(
        model="tiny",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        stream=True,
    )
    next(stream)
    client.close()
    with pytest.raises(RequestCancelledError):
        list(stream)
