from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


def dynamic_chunk_size(
    initial_size: int,
    *,
    prefix_tokens: int,
    smoothing_factor: float,
    page_size: int = 64,
) -> int:
    if initial_size < 1 or prefix_tokens < 0 or page_size < 1:
        raise ValueError("chunk, prefix, and page sizes must be nonnegative and positive")
    if not 0 <= smoothing_factor <= 1:
        raise ValueError("smoothing_factor must be between zero and one")
    reduction = 1 - smoothing_factor * prefix_tokens / (prefix_tokens + initial_size)
    target = max(page_size, int(initial_size * reduction))
    return max(page_size, target // page_size * page_size)


@dataclass(frozen=True)
class ScheduledChunk:
    request_id: str
    tokens: int


class ContinuousBatchScheduler:
    def __init__(self, *, max_running: int, chunk_tokens: int = 64) -> None:
        if max_running < 1 or chunk_tokens < 1:
            raise ValueError("max_running and chunk_tokens must be positive")
        self.max_running = max_running
        self.chunk_tokens = chunk_tokens
        self._pending: deque[tuple[str, int]] = deque()
        self._running: dict[str, int] = {}
        self._lock = threading.RLock()

    def submit(self, request_id: str, prompt_tokens: int) -> None:
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        with self._lock:
            if request_id in self._running or any(request_id == item[0] for item in self._pending):
                raise ValueError(f"duplicate request {request_id}")
            self._pending.append((request_id, prompt_tokens))

    def next_batch(self) -> list[ScheduledChunk]:
        with self._lock:
            while len(self._running) < self.max_running and self._pending:
                request_id, tokens = self._pending.popleft()
                self._running[request_id] = tokens
            batch = [
                ScheduledChunk(request_id, min(self.chunk_tokens, tokens))
                for request_id, tokens in self._running.items()
            ]
            for chunk in batch:
                self._running[chunk.request_id] -= chunk.tokens
            return batch

    def complete(self, request_id: str) -> None:
        with self._lock:
            self._running.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._running.pop(request_id, None)
            self._pending = deque(
                (pending_id, tokens)
                for pending_id, tokens in self._pending
                if pending_id != request_id
            )
