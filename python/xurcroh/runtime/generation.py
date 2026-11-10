from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event
from typing import Literal

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


@dataclass(frozen=True)
class GenerationEvent:
    delta: str
    summary: GenerationSummary | None = None


def sample_next_token(logits: torch.Tensor, request: CompletionRequest) -> torch.Tensor:
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
    generator = None if request.seed is None else torch.Generator().manual_seed(request.seed)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


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


def iter_completion_text(text: str) -> Iterator[str]:
    yield text
