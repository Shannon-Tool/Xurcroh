from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from typing import Any, Literal

import torch

from xurcroh.errors import RequestCancelledError, RequestTimeoutError
from xurcroh.requests import CompletionRequest


@dataclass(frozen=True)
class GenerationSummary:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    duration_ms: float
    seed: int | None = None


@dataclass(frozen=True)
class GenerationEvent:
    delta: str
    summary: GenerationSummary | None = None


def sample_next_token(
    logits: torch.Tensor,
    request: CompletionRequest,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if request.temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / request.temperature
    if request.top_k > 0:
        count = min(request.top_k, scaled.shape[-1])
        threshold = torch.topk(scaled, count).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
    if request.top_p < 1:
        sorted_logits, indices = torch.sort(scaled, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities > request.top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(-1, indices, sorted_logits)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def effective_seed(request: CompletionRequest) -> int | None:
    """Return the seed that drives sampling, or ``None`` for greedy decoding."""

    if request.temperature == 0:
        return None
    return request.seed if request.seed is not None else secrets.randbits(63)


class TokenSampler:
    """One request's sampler: a single generator that advances across tokens."""

    def __init__(self, request: CompletionRequest) -> None:
        self._request = request
        self.seed = effective_seed(request)
        self._generator = None if self.seed is None else torch.Generator().manual_seed(self.seed)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return sample_next_token(logits, self._request, generator=self._generator)


def normalize_stop_token_ids(*values: object) -> set[int]:
    normalized: set[int] = set()
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, list | tuple | set) else [value]
        for item in items:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError("stop token IDs must be integers")
            normalized.add(item)
    return normalized


def stopping_strings(request: CompletionRequest) -> tuple[str, ...]:
    if request.stop is None:
        return ()
    return (request.stop,) if isinstance(request.stop, str) else tuple(request.stop)


def check_request_health(cancelled: Event, started: float, timeout: float) -> None:
    import time

    if bool(cancelled.is_set()):
        raise RequestCancelledError("request was cancelled")
    if time.monotonic() - started > timeout:
        raise RequestTimeoutError(f"request exceeded {timeout:.3f}s timeout")


@contextmanager
def bounded_execution(
    lock: Any,
    cancelled: Event,
    started: float,
    timeout: float,
    model: str,
) -> Iterator[None]:
    import time

    check_request_health(cancelled, started, timeout)
    remaining = timeout - (time.monotonic() - started)
    if not lock.acquire(timeout=max(remaining, 0)):
        raise RequestTimeoutError(f"model {model!r} inference queue exceeded {timeout:.6f}s")
    try:
        yield
    finally:
        lock.release()


def iter_completion_text(text: str) -> Iterator[str]:
    yield text
