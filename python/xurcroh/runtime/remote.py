from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any, cast

import httpx

from xurcroh.config import SecurityConfig, WorkerConfig
from xurcroh.errors import (
    CapabilityError,
    ClusterUnavailableError,
    RequestCancelledError,
)
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.generation import GenerationEvent, GenerationSummary, check_request_health
from xurcroh.types import ChatCompletion, ChatCompletionChunk


class RemoteOpenAIEngine:
    def __init__(self, workers: list[WorkerConfig], security: SecurityConfig) -> None:
        if not workers:
            raise ClusterUnavailableError("remote execution requires at least one worker")
        self.workers = workers
        self.security = security
        self.engine_name = "remote-openai"
        self.worker_id = workers[0].name
        self._endpoints = tuple(dict.fromkeys(self._endpoint(worker) for worker in workers))
        self._client = httpx.Client(timeout=60)
        self._closed = False

    @staticmethod
    def _endpoint(worker: WorkerConfig) -> str:
        endpoint = worker.endpoint or ""
        if endpoint.startswith("http+dev://"):
            return "http://" + endpoint.removeprefix("http+dev://")
        return endpoint.rstrip("/")

    @classmethod
    def validate_worker(
        cls, worker: WorkerConfig, security: SecurityConfig, *, stage: bool = False
    ) -> None:
        endpoint = worker.endpoint or ""
        if security.mode != "development" and not stage:
            raise ClusterUnavailableError(
                "strict remote execution requires a compiled stage manifest; "
                "full-model workers require explicit development policy"
            )
        if endpoint.startswith("http+dev://") and security.mode != "development":
            raise ClusterUnavailableError("strict mode rejects development HTTP worker endpoints")
        if endpoint.startswith("http://"):
            raise ClusterUnavailableError("strict mode requires HTTPS or QUIC worker endpoints")
        if endpoint.startswith("quic://"):
            if stage:
                return
            raise ClusterUnavailableError(
                "QUIC stage transport requires a stage runner, not an OpenAI endpoint"
            )
        if not endpoint.startswith("https://") and not endpoint.startswith("http+dev://"):
            raise ClusterUnavailableError(f"unsupported remote worker endpoint: {endpoint}")

    @property
    def modality(self) -> str:
        return self.workers[0].labels.get("modality", "text")

    def require_modality(self, expected: str) -> None:
        if self.modality != expected:
            raise CapabilityError(f"remote worker does not support {expected} generation")

    @staticmethod
    def _headers(worker: WorkerConfig) -> dict[str, str]:
        if worker.credential_env is None:
            return {}
        import os

        credential = os.environ.get(worker.credential_env)
        if credential is None:
            raise ClusterUnavailableError(
                f"worker credential variable {worker.credential_env} is not set"
            )
        return {"Authorization": f"Bearer {credential}"}

    def _body(self, request: CompletionRequest) -> dict[str, Any]:
        return request.model_dump(mode="json", exclude_none=True)

    def _post(self, request: CompletionRequest) -> ChatCompletion:
        errors = []
        for worker in self.workers:
            try:
                response = self._client.post(
                    f"{self._endpoint(worker)}/v1/chat/completions",
                    json=self._body(request),
                    headers=self._headers(worker),
                    timeout=request.timeout,
                )
                response.raise_for_status()
                return ChatCompletion.model_validate(response.json())
            except (httpx.HTTPError, ValueError) as error:
                errors.append(f"{worker.name}: {type(error).__name__}")
        raise ClusterUnavailableError("all remote workers failed: " + ", ".join(errors))

    def _stream(self, request: CompletionRequest) -> Iterator[ChatCompletionChunk]:
        errors = []
        for worker in self.workers:
            emitted = False
            try:
                with self._client.stream(
                    "POST",
                    f"{self._endpoint(worker)}/v1/chat/completions",
                    json=self._body(request),
                    headers=self._headers(worker),
                    timeout=request.timeout,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = line.removeprefix("data: ") if line.startswith("data:") else line
                        if data == "[DONE]":
                            continue
                        emitted = True
                        yield ChatCompletionChunk.model_validate(json.loads(data))
                    return
            except (httpx.HTTPError, ValueError) as error:
                if emitted:
                    raise ClusterUnavailableError(
                        f"remote stream from {worker.name} failed after partial output; "
                        "retry was suppressed to avoid duplicate tokens"
                    ) from error
                errors.append(f"{worker.name}: {type(error).__name__}")
        if errors:
            raise ClusterUnavailableError("all remote workers failed: " + ", ".join(errors))

    def generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        started = time.monotonic()
        if request.stream:
            summary = None
            for chunk in self._stream(request):
                check_request_health(cancelled, started, timeout)
                stream_choice = chunk.choices[0]
                if stream_choice.finish_reason is None:
                    yield GenerationEvent(delta=stream_choice.delta.content or "")
                else:
                    if stream_choice.finish_reason == "cancelled":
                        raise RequestCancelledError("remote request was cancelled")
                    usage = chunk.usage
                    if usage is None or chunk.execution is None:
                        raise ClusterUnavailableError("remote final chunk lacks usage metadata")
                    summary = GenerationSummary(
                        text="",
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        finish_reason=stream_choice.finish_reason,
                        duration_ms=chunk.execution.duration_ms,
                    )
                    yield GenerationEvent(delta="", summary=summary)
            if summary is None:
                raise ClusterUnavailableError("remote stream ended without a final chunk")
        else:
            check_request_health(cancelled, started, timeout)
            response = self._post(request)
            response_choice = response.choices[0]
            if response_choice.finish_reason == "cancelled":
                raise RequestCancelledError("remote request was cancelled")
            yield GenerationEvent(delta=response_choice.message.content)
            yield GenerationEvent(
                delta="",
                summary=GenerationSummary(
                    text=response_choice.message.content,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    finish_reason=response_choice.finish_reason,
                    duration_ms=response.execution.duration_ms,
                ),
            )

    def health(self) -> dict[str, Any]:
        errors = []
        for worker in self.workers:
            try:
                response = self._client.get(
                    f"{self._endpoint(worker)}/health",
                    headers=self._headers(worker),
                )
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except httpx.HTTPError as error:
                errors.append(f"{worker.name}: {type(error).__name__}")
        raise ClusterUnavailableError("no healthy remote workers: " + ", ".join(errors))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()
