from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import AsyncXurcroh, Xurcroh
from xurcroh.errors import CapabilityError, ClientClosedError, ModelNotFoundError


def make_client(path: Path) -> Xurcroh:
    return Xurcroh(cluster={"endpoint": "local://cpu"}, model_path={"qwen-7b": path})


def test_exact_public_import_and_real_chat_completion(tiny_model_path: Path) -> None:
    with make_client(tiny_model_path) as client:
        response = client.chat.completions.create(
            model="qwen-7b",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
            temperature=0,
            max_tokens=3,
        )
        assert response.object == "chat.completion"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content is not None
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )
        assert response.execution.engine == "transformers"
        assert response.execution.duration_ms > 0


def test_streaming_returns_real_token_deltas(tiny_model_path: Path) -> None:
    with make_client(tiny_model_path) as client:
        stream = client.chat.completions.create(
            model="qwen-7b",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=3,
            stream=True,
        )
        chunks = list(stream)
        assert chunks
        assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
        assert any(chunk.choices[0].delta.content for chunk in chunks)
        assert chunks[-1].choices[0].finish_reason is not None


def test_unknown_model_and_unsupported_modality_fail_cleanly(tiny_model_path: Path) -> None:
    with make_client(tiny_model_path) as client:
        with pytest.raises(ModelNotFoundError, match="available"):
            client.chat.completions.create(
                model="missing", messages=[{"role": "user", "content": "hello"}]
            )
        with pytest.raises(CapabilityError, match="does not support image"):
            client.images.generate(model="qwen-7b", prompt="city")


def test_closed_client_rejects_requests(tiny_model_path: Path) -> None:
    client = make_client(tiny_model_path)
    client.close()
    with pytest.raises(ClientClosedError):
        client.chat.completions.create(
            model="qwen-7b", messages=[{"role": "user", "content": "hello"}]
        )


@pytest.mark.asyncio
async def test_async_completion_and_streaming(tiny_model_path: Path) -> None:
    async with AsyncXurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"qwen-7b": tiny_model_path}
    ) as client:
        response = await client.chat.completions.create(
            model="qwen-7b",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=2,
        )
        assert response.usage.completion_tokens > 0
        stream = await client.chat.completions.create(
            model="qwen-7b",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=2,
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        assert chunks


def test_response_types_are_publicly_importable() -> None:
    from Xurcroh import ChatCompletion, ChatCompletionChunk, CompletionUsage

    assert ChatCompletion.__name__ == "ChatCompletion"
    assert ChatCompletionChunk.__name__ == "ChatCompletionChunk"
    assert CompletionUsage.__name__ == "CompletionUsage"
