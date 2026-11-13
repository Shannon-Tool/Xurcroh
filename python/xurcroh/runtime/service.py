from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from xurcroh.config import ClusterConfig, ModelRegistry, WorkerConfig
from xurcroh.errors import (
    CapabilityError,
    ClientClosedError,
    ClusterUnavailableError,
    ConfigurationError,
    RequestCancelledError,
    RequestTimeoutError,
)
from xurcroh.fortress.engine import FortressEngine
from xurcroh.provisioning import provision_cluster_sync
from xurcroh.requests import CompletionRequest, ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.diffusion import LocalDiffusionEngine
from xurcroh.runtime.dit_stage import CompiledDiTStageEngine
from xurcroh.runtime.local import LocalTransformersEngine
from xurcroh.runtime.media import RemoteMediaEngine
from xurcroh.runtime.media_protected import RemoteProtectedMediaEngine
from xurcroh.runtime.metrics import RuntimeMetrics
from xurcroh.runtime.remote import RemoteOpenAIEngine
from xurcroh.runtime.stage import DenseStageEngine
from xurcroh.runtime.stream import ManagedStream
from xurcroh.runtime.vlm import LocalVisionLanguageEngine
from xurcroh.types import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatDelta,
    ChatMessage,
    CompletionUsage,
    ExecutionMetadata,
    ImageGenerationResponse,
    VideoGenerationResponse,
)


@dataclass
class RequestHandle:
    request_id: str
    cancelled: threading.Event
    model: str


class RuntimeService:
    def __init__(self, cluster: ClusterConfig, models: ModelRegistry) -> None:
        self.cluster = cluster
        self.models = models
        self._engines: dict[
            str,
            LocalTransformersEngine
            | RemoteOpenAIEngine
            | LocalDiffusionEngine
            | CompiledDiTStageEngine
            | DenseStageEngine
            | LocalVisionLanguageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine
            | FortressEngine,
        ] = {}
        self._requests: dict[str, RequestHandle] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._metrics = RuntimeMetrics()
        if cluster.provision_workers:
            if cluster.controller_endpoint is None:
                raise ConfigurationError("controller_endpoint is required for provisioning")
            provision_cluster_sync(
                cluster,
                models.items(),
                controller=cluster.controller_endpoint,
            )
        stage_workers = any(
            definition.manifest_path is not None or definition.engine == "fortress"
            for _, definition in models.items()
        )
        for worker in cluster.workers:
            if worker.endpoint != "local://cpu":
                RemoteOpenAIEngine.validate_worker(worker, cluster.security, stage=stage_workers)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClientClosedError("Xurcroh client is closed")

    def _workers_for_model(self, name: str) -> list[WorkerConfig]:
        workers = [
            worker for worker in self.cluster.workers if not worker.models or name in worker.models
        ]
        if not workers:
            raise ClusterUnavailableError(f"no configured worker advertises model {name!r}")
        return workers

    def _engine(
        self, name: str
    ) -> (
        LocalTransformersEngine
        | RemoteOpenAIEngine
        | LocalDiffusionEngine
        | CompiledDiTStageEngine
        | DenseStageEngine
        | LocalVisionLanguageEngine
        | RemoteMediaEngine
        | RemoteProtectedMediaEngine
        | FortressEngine
    ):
        definition = self.models.get(name)
        workers = self._workers_for_model(name)
        with self._lock:
            self._ensure_open()
            engine = self._engines.get(name)
            if engine is None:
                if definition.engine == "fortress":
                    engine = FortressEngine(name, definition, workers)
                elif definition.manifest_path is not None and definition.modality in {
                    "image",
                    "video",
                }:
                    engine = RemoteProtectedMediaEngine(
                        name, definition, workers, self.cluster.security
                    )
                elif definition.manifest_path is not None:
                    engine = DenseStageEngine(
                        name,
                        definition,
                        workers,
                        self.cluster.security,
                        self.cluster.controller_endpoint,
                    )
                elif workers[0].endpoint == "local://cpu":
                    probe = LocalTransformersEngine(name, definition, workers[0].name)
                    if probe.modality in {"image", "video"}:
                        engine = LocalDiffusionEngine(name, definition, workers[0].name)
                    elif probe.modality == "vlm":
                        engine = LocalVisionLanguageEngine(name, definition, workers[0].name)
                    else:
                        engine = probe
                elif definition.protect_latents:
                    engine = RemoteProtectedMediaEngine(
                        name, definition, workers, self.cluster.security
                    )
                elif definition.modality == "image":
                    engine = RemoteMediaEngine(workers, self.cluster.security, "image")
                elif definition.modality == "video":
                    engine = RemoteMediaEngine(workers, self.cluster.security, "video")
                else:
                    engine = RemoteOpenAIEngine(workers, self.cluster.security)
                self._engines[name] = engine
            return engine

    def _register(
        self,
        request: CompletionRequest | ImageGenerationRequest | VideoGenerationRequest,
    ) -> RequestHandle:
        request_id = request.request_id or f"req_{uuid.uuid4().hex}"
        handle = RequestHandle(
            request_id=request_id,
            cancelled=threading.Event(),
            model=request.model,
        )
        with self._lock:
            self._ensure_open()
            if request_id in self._requests:
                raise ConfigurationError(f"request_id {request_id!r} is already active")
            self._requests[request_id] = handle
        return handle

    def _admit(self, model: str, timeout: float) -> None:
        capacity = min(
            worker.resources.max_concurrent_requests for worker in self._workers_for_model(model)
        )
        with self._lock:
            semaphore = self._semaphores.get(model)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(capacity)
                self._semaphores[model] = semaphore
        if not semaphore.acquire(timeout=timeout):
            raise RequestTimeoutError(f"admission timeout after {timeout:.6f}s")
        self._metrics.started()

    def _release(self, model: str) -> None:
        self._semaphores[model].release()

    def _record_error(self, error: BaseException) -> None:
        if isinstance(error, RequestTimeoutError):
            self._metrics.timed_out()
        elif isinstance(error, RequestCancelledError):
            self._metrics.cancelled()
        else:
            self._metrics.failed()

    def metrics(self) -> dict[str, int | float]:
        return self._metrics.snapshot()

    def _metadata(
        self,
        request_id: str,
        duration_ms: float,
        engine_name: str,
        worker_ids: list[str],
    ) -> ExecutionMetadata:
        return ExecutionMetadata(
            request_id=request_id,
            engine=engine_name,
            worker_ids=worker_ids,
            duration_ms=max(duration_ms, 0.001),
            security_mode=self.cluster.security.mode,
        )

    def complete(self, request: CompletionRequest) -> ChatCompletion:
        self._ensure_open()
        engine = self._engine(request.model)
        if isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine,
        ):
            raise CapabilityError("diffusion engines do not expose text chat completions")
        handle = self._register(request)
        created = int(time.time())
        timeout = request.timeout or self.cluster.request_timeout_seconds
        admitted = False
        try:
            self._admit(request.model, timeout)
            admitted = True
            summary = None
            for event in engine.generate(request, cancelled=handle.cancelled, timeout=timeout):
                if event.summary is not None:
                    summary = event.summary
            if summary is None:
                raise RuntimeError("generation ended without a summary")
            self._metrics.completed(summary.duration_ms)
            return ChatCompletion(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=summary.text),
                        finish_reason=summary.finish_reason,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=summary.prompt_tokens,
                    completion_tokens=summary.completion_tokens,
                    total_tokens=summary.prompt_tokens + summary.completion_tokens,
                ),
                execution=self._metadata(
                    handle.request_id,
                    summary.duration_ms,
                    engine.engine_name,
                    getattr(engine, "worker_ids", [engine.worker_id]),
                ),
            )
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)

    def stream(self, request: CompletionRequest) -> Iterator[ChatCompletionChunk]:
        self._ensure_open()
        engine = self._engine(request.model)
        if isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine,
        ):
            raise CapabilityError("diffusion engines do not expose text chat completions")
        handle = self._register(request)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        timeout = request.timeout or self.cluster.request_timeout_seconds
        try:
            self._admit(request.model, timeout)
        except Exception as error:
            self._record_error(error)
            with self._lock:
                self._requests.pop(handle.request_id, None)
            raise

        def cleanup() -> None:
            self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)

        def generate() -> Iterator[ChatCompletionChunk]:
            try:
                for event in engine.generate(request, cancelled=handle.cancelled, timeout=timeout):
                    if event.summary is None:
                        yield ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatChunkChoice(
                                    index=0, delta=ChatDelta(role="assistant", content=event.delta)
                                )
                            ],
                        )
                    else:
                        self._metrics.completed(event.summary.duration_ms)
                        yield ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatChunkChoice(
                                    index=0,
                                    delta=ChatDelta(),
                                    finish_reason=event.summary.finish_reason,
                                )
                            ],
                            usage=CompletionUsage(
                                prompt_tokens=event.summary.prompt_tokens,
                                completion_tokens=event.summary.completion_tokens,
                                total_tokens=event.summary.prompt_tokens
                                + event.summary.completion_tokens,
                            ),
                            execution=self._metadata(
                                handle.request_id,
                                event.summary.duration_ms,
                                engine.engine_name,
                                getattr(engine, "worker_ids", [engine.worker_id]),
                            ),
                        )
            except Exception as error:
                self._record_error(error)
                raise

        return ManagedStream(generate(), cleanup)

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        self._ensure_open()
        engine = self._engine(request.model)
        engine.require_modality("image")
        if not isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine,
        ):
            raise ClusterUnavailableError(
                f"model {request.model!r} requires an initialized diffusion worker"
            )
        handle = self._register(request)
        timeout = request.timeout or self.cluster.request_timeout_seconds
        admitted = False
        try:
            self._admit(request.model, timeout)
            admitted = True
            response = engine.generate_image(
                request,
                request_id=handle.request_id,
                security_mode=self.cluster.security.mode,
                cancelled=handle.cancelled,
                timeout=timeout,
            )
            self._metrics.completed(response.execution.duration_ms)
            return response
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)

    def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        self._ensure_open()
        engine = self._engine(request.model)
        engine.require_modality("video")
        if not isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine,
        ):
            raise ClusterUnavailableError(
                f"model {request.model!r} requires an initialized video worker"
            )
        handle = self._register(request)
        timeout = request.timeout or self.cluster.request_timeout_seconds
        admitted = False
        try:
            self._admit(request.model, timeout)
            admitted = True
            response = engine.generate_video(
                request,
                request_id=handle.request_id,
                security_mode=self.cluster.security.mode,
                cancelled=handle.cancelled,
                timeout=timeout,
            )
            self._metrics.completed(response.execution.duration_ms)
            return response
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            handle = self._requests.get(request_id)
            if handle is None:
                return False
            handle.cancelled.set()
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for handle in self._requests.values():
                handle.cancelled.set()
            engines = list(self._engines.values())
            self._engines.clear()
            self._requests.clear()
        for engine in engines:
            engine.close()
