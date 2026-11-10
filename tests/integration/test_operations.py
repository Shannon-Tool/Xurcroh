from __future__ import annotations

import httpx

from Xurcroh import Xurcroh


def test_worker_stream_uses_openai_server_sent_events(running_worker: str) -> None:
    endpoint = running_worker.replace("http+dev://", "http://")
    with httpx.stream(
        "POST",
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
        },
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "data: {" in body
    assert "data: [DONE]" in body


def test_worker_metrics_endpoint_reports_only_aggregate_counters(running_worker: str) -> None:
    endpoint = running_worker.replace("http+dev://", "http://")
    response = httpx.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "private prompt"}],
            "max_tokens": 1,
        },
    )
    response.raise_for_status()
    metrics_response = httpx.get(f"{endpoint}/metrics")
    metrics_response.raise_for_status()
    metrics = metrics_response.json()
    assert metrics["requests_started"] == 1
    assert metrics["requests_completed"] == 1
    assert "private prompt" not in metrics_response.text


def test_public_metrics_exclude_request_content(tiny_model_path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "sensitive benchmark prompt"}],
            max_tokens=1,
        )
        metrics = client.metrics()
    assert metrics["requests_started"] == 1
    assert metrics["requests_completed"] == 1
    assert "sensitive benchmark prompt" not in repr(metrics)
