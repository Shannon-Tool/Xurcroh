from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

from Xurcroh._client import Xurcroh
from xurcroh.types import ChatCompletion, ChatCompletionChunk


class AsyncChatCompletions:
    def __init__(self, client: AsyncXurcroh) -> None:
        self._client = client

    async def create(
        self, **parameters: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        result = await asyncio.to_thread(self._client._sync.chat.completions.create, **parameters)
        if isinstance(result, ChatCompletion):
            return result

        async def iterate() -> AsyncIterator[ChatCompletionChunk]:
            iterator = iter(result)
            sentinel = object()
            try:
                while True:
                    chunk = await asyncio.to_thread(next, iterator, sentinel)
                    if chunk is sentinel:
                        break
                    yield cast(ChatCompletionChunk, chunk)
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    await asyncio.to_thread(close)

        return iterate()


class AsyncChat:
    def __init__(self, client: AsyncXurcroh) -> None:
        self.completions = AsyncChatCompletions(client)


class AsyncImages:
    def __init__(self, client: AsyncXurcroh) -> None:
        self._client = client

    async def generate(self, **parameters: Any) -> Any:
        return await asyncio.to_thread(self._client._sync.images.generate, **parameters)

    async def edit(self, **parameters: Any) -> Any:
        return await asyncio.to_thread(self._client._sync.images.edit, **parameters)


class AsyncVideos:
    def __init__(self, client: AsyncXurcroh) -> None:
        self._client = client

    async def generate(self, **parameters: Any) -> Any:
        return await asyncio.to_thread(self._client._sync.videos.generate, **parameters)


class AsyncXurcroh:
    def __init__(self, **parameters: Any) -> None:
        self._sync = Xurcroh(**parameters)
        self.chat = AsyncChat(self)
        self.images = AsyncImages(self)
        self.videos = AsyncVideos(self)

    async def cancel(self, request_id: str) -> bool:
        return await asyncio.to_thread(self._sync.cancel, request_id)

    async def metrics(self) -> dict[str, int | float]:
        return await asyncio.to_thread(self._sync.metrics)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def __aenter__(self) -> AsyncXurcroh:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
