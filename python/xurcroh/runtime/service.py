from __future__ import annotations

import sys
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xurcroh import __version__
from xurcroh.config import ClusterConfig, ModelRegistry, WorkerConfig
from xurcroh.errors import (
    CapabilityError,
    ClientClosedError,
    ClusterUnavailableError,
    ConfigurationError,
    RequestCancelledError,
    RequestTimeoutError,
)
from xurcroh.fortress.diffusion_engine import FortressDiffusionEngine
from xurcroh.fortress.engine import FortressEngine
from xurcroh.fortress.prompt_armour import ArmouredMessages, MessageArmour, PromptArmour
from xurcroh.memory_host import protect_process
from xurcroh.preflight import CheckpointPreflight, preflight_checkpoint
from xurcroh.provisioning import provision_cluster_sync
from xurcroh.receipts import SAMPLER, TraceStore, canonical_media, canonical_messages, commit
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

if TYPE_CHECKING:
    from xurcroh.fortress.deployment import FortressDeployment
    from xurcroh.runtime.generation import GenerationSummary


@dataclass
class RequestHandle:
    request_id: str
    cancelled: threading.Event
    model: str
    executing: threading.Event
    finished: threading.Event


class RuntimeService:
    def __init__(self, cluster: ClusterConfig, models: ModelRegistry) -> None:
        self.cluster = cluster
        self.models = models
        if cluster.security.process_hardening and sys.platform == "linux":
            protect_process()
        self._traces = TraceStore(cluster.security.trace_history)
        self._model_facts: dict[str, dict[str, Any]] = {}
        self._message_armour: MessageArmour | None = None
        if cluster.security.prompt_armour:
            self._message_armour = MessageArmour(
                PromptArmour(model_path=cluster.security.prompt_armour_model)
            )
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
            | FortressEngine
            | FortressDiffusionEngine,
        ] = {}
        self._requests: dict[str, RequestHandle] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._metrics = RuntimeMetrics()
        self._fortress_metrics: dict[str, int] = {}
        self._deployment: FortressDeployment | None = None
        self._provisioned_workers: dict[str, list[WorkerConfig]] = {}
        if cluster.provision_workers:
            self._deployment = provision_cluster_sync(
                cluster,
                models.items(),
                controller=cluster.controller_endpoint or "sdk-managed",
            )
            self._provisioned_workers = {
                name: list(workers) for name, workers in self._deployment.workers.items()
            }
        stage_workers = any(
            definition.manifest_path is not None or definition.engine == "fortress"
            for _, definition in models.items()
        )
        for worker in cluster.workers:
            if not cluster.provision_workers and worker.endpoint != "local://cpu":
                RemoteOpenAIEngine.validate_worker(worker, cluster.security, stage=stage_workers)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClientClosedError("Xurcroh client is closed")

    def _workers_for_model(self, name: str) -> list[WorkerConfig]:
        if name in self._provisioned_workers:
            return self._provisioned_workers[name]
        workers = [
            worker for worker in self.cluster.workers if not worker.models or name in worker.models
        ]
        if not workers:
            raise ClusterUnavailableError(f"no configured worker advertises model {name!r}")
        return workers

    def preflight(self, model: str | None = None) -> dict[str, CheckpointPreflight]:
        self._ensure_open()
        names = (model,) if model is not None else self.models.names()
        reports = {
            name: preflight_checkpoint(name, self.models.get(name), self._workers_for_model(name))
            for name in names
        }
        return reports

    def _require_preflight(self, name: str) -> None:
        report = preflight_checkpoint(
            name,
            self.models.get(name),
            self._workers_for_model(name),
            require_known_memory=self.cluster.provision_workers,
        )
        if not report.supported:
            raise ConfigurationError(
                f"model {name!r} preflight failed: " + "; ".join(report.issues)
            )

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
        | FortressDiffusionEngine
    ):
        definition = self.models.get(name)
        workers = self._workers_for_model(name)
        if definition.source_format == "auto" and (
            definition.engine == "fortress" or Path(definition.path).expanduser().exists()
        ):
            from xurcroh.preflight import detect_modality

            detect_modality(definition)
        if definition.engine == "fortress":
            self._require_preflight(name)
        with self._lock:
            self._ensure_open()
            engine = self._engines.get(name)
            if engine is None:
                if definition.engine == "fortress":
                    if definition.modality in {"image", "video"}:
                        engine = FortressDiffusionEngine(
                            name,
                            definition,
                            workers,
                            recover_role=(
                                self._deployment.restart_role
                                if self._deployment is not None
                                else None
                            ),
                        )
                    else:
                        engine = FortressEngine(
                            name,
                            definition,
                            workers,
                            recover_role=(
                                self._deployment.restart_role
                                if self._deployment is not None
                                else None
                            ),
                        )
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
            executing=threading.Event(),
            finished=threading.Event(),
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
        if self._deployment is not None:
            for role, report in self._deployment.diagnostics().items():
                error.add_note(f"{role}: {report}")

    def metrics(self) -> dict[str, int | float]:
        snapshot = self._metrics.snapshot()
        snapshot.update(self._fortress_metrics)
        for engine in self._engines.values():
            if isinstance(engine, FortressEngine):
                snapshot.update(engine.transport_metrics())
        return snapshot

    def _metadata(
        self,
        request_id: str,
        duration_ms: float,
        engine_name: str,
        worker_ids: list[str],
        *,
        seed: int | None = None,
        armoured: ArmouredMessages | None = None,
    ) -> ExecutionMetadata:
        return ExecutionMetadata(
            seed=seed,
            armoured_entities=None if armoured is None else armoured.entities,
            request_id=request_id,
            engine=engine_name,
            worker_ids=worker_ids,
            duration_ms=max(duration_ms, 0.001),
            security_mode=self.cluster.security.mode,
        )

    def trace(self, task_id: str) -> dict[str, Any]:
        return self._traces.get(task_id)

    @property
    def trace_public_key(self) -> bytes:
        return self._traces.public_key

    def _facts(self, name: str, engine: Any) -> dict[str, Any]:
        with self._lock:
            facts = self._model_facts.get(name)
        if facts is None:
            definition = self.models.get(name)
            report = preflight_checkpoint(name, definition, self._workers_for_model(name))
            facts = {
                "name": name,
                "revision": definition.revision,
                "revision_pinned": report.revision_pinned,
                "source_format": report.source_format,
                "source_digest": report.source_digest,
                "layout_digest": report.checkpoint_layout_digest,
                "architecture": report.architecture,
                "encodings": list(report.encodings),
                "engine": engine.engine_name,
            }
            with self._lock:
                self._model_facts[name] = facts
        return facts

    def _security_facts(self) -> dict[str, Any]:
        security = self.cluster.security
        return {
            "mode": security.mode,
            "attestation": "required" if security.require_attestation else "unattested",
            "process_hardening": security.process_hardening,
            "prompt_armour": security.prompt_armour,
            "memory_scrubbing": security.memory_scrubbing,
        }

    def _record_media_trace(
        self,
        kind: str,
        request: ImageGenerationRequest | VideoGenerationRequest,
        response: ImageGenerationResponse | VideoGenerationResponse,
        started_at: float,
    ) -> None:
        engine = self._engine(request.model)
        execution = response.execution
        self._traces.record(
            {
                "task_id": execution.request_id,
                "mode": "local",
                "created_at": started_at,
                "completed_at": time.time(),
                "client": {"version": __version__},
                "request": {
                    "kind": kind,
                    "settings": request.model_dump(
                        mode="json", exclude={"prompt", "image", "mask", "request_id"}
                    ),
                    "effective": {"seed": execution.seed},
                    "prompt": commit(
                        canonical_media(
                            {
                                "prompt": request.prompt,
                                "image": request.image,
                                "mask": getattr(request, "mask", None),
                            }
                        )
                    ),
                },
                "result": {
                    "output": commit("\n".join(item.b64_json for item in response.data).encode()),
                    "items": len(response.data),
                    "duration_ms": execution.duration_ms,
                },
                "model": self._facts(request.model, engine),
                "security": self._security_facts(),
                "transport": {"workers": execution.worker_ids, "counters": None},
            }
        )

    def _record_trace(
        self,
        task_id: str,
        request: CompletionRequest,
        summary: GenerationSummary,
        output: str,
        started_at: float,
        armoured: ArmouredMessages | None,
    ) -> None:
        engine = self._engine(request.model)
        split = engine.trace_facts() if isinstance(engine, FortressEngine) else {}
        self._traces.record(
            {
                "task_id": task_id,
                "mode": "local",
                "created_at": started_at,
                "completed_at": time.time(),
                "client": {"version": __version__},
                "request": {
                    "kind": "chat",
                    "settings": request.model_dump(mode="json", exclude={"messages", "request_id"}),
                    "effective": {
                        "seed": summary.seed,
                        "sampler": SAMPLER,
                        "armoured_entities": None if armoured is None else armoured.entities,
                    },
                    "prompt": commit(canonical_messages(request.messages)),
                },
                "result": {
                    "output": commit(output.encode()),
                    "finish_reason": summary.finish_reason,
                    "prompt_tokens": summary.prompt_tokens,
                    "completion_tokens": summary.completion_tokens,
                    "duration_ms": summary.duration_ms,
                },
                "model": {**self._facts(request.model, engine), **split.get("sharing", {})},
                "security": self._security_facts(),
                "transport": {
                    "workers": getattr(engine, "worker_ids", [engine.worker_id]),
                    **split.get("transport", {}),
                    "counters": summary.transport,
                },
            }
        )

    def _armour(
        self, request: CompletionRequest
    ) -> tuple[CompletionRequest, ArmouredMessages | None]:
        if self._message_armour is None:
            return request, None
        armoured = self._message_armour.protect(request.messages)
        return request.model_copy(update={"messages": armoured.messages}), armoured

    def complete(self, request: CompletionRequest) -> ChatCompletion:
        self._ensure_open()
        engine = self._engine(request.model)
        if isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine
            | FortressDiffusionEngine,
        ):
            raise CapabilityError("diffusion engines do not expose text chat completions")
        handle = self._register(request)
        created = int(time.time())
        timeout = request.timeout or self.cluster.request_timeout_seconds
        admitted = False
        armoured: ArmouredMessages | None = None
        submitted = request
        started_at = time.time()
        try:
            request, armoured = self._armour(request)
            self._admit(request.model, timeout)
            admitted = True
            handle.executing.set()
            summary = None
            for event in engine.generate(request, cancelled=handle.cancelled, timeout=timeout):
                if event.summary is not None:
                    summary = event.summary
            if summary is None:
                raise RuntimeError("generation ended without a summary")
            self._metrics.completed(summary.duration_ms)
            text = summary.text if armoured is None else armoured.restore(summary.text)
            self._record_trace(handle.request_id, submitted, summary, text, started_at, armoured)
            return ChatCompletion(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
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
                    seed=summary.seed,
                    armoured=armoured,
                ),
            )
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            if armoured is not None:
                armoured.close()
            handle.executing.clear()
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)
            handle.finished.set()

    def stream(self, request: CompletionRequest) -> Iterator[ChatCompletionChunk]:
        self._ensure_open()
        engine = self._engine(request.model)
        if isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine
            | FortressDiffusionEngine,
        ):
            raise CapabilityError("diffusion engines do not expose text chat completions")
        handle = self._register(request)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        timeout = request.timeout or self.cluster.request_timeout_seconds
        armoured: ArmouredMessages | None = None
        submitted = request
        started_at = time.time()
        try:
            request, armoured = self._armour(request)
            self._admit(request.model, timeout)
        except Exception as error:
            self._record_error(error)
            if armoured is not None:
                armoured.close()
            with self._lock:
                self._requests.pop(handle.request_id, None)
            raise
        restorer = None if armoured is None else armoured.stream()
        emitted: list[str] = []

        def chunk(content: str) -> ChatCompletionChunk:
            emitted.append(content)
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatChunkChoice(index=0, delta=ChatDelta(role="assistant", content=content))
                ],
            )

        def cleanup() -> None:
            if armoured is not None:
                armoured.close()
            handle.executing.clear()
            self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)
            handle.finished.set()

        def generate() -> Iterator[ChatCompletionChunk]:
            iterator = engine.generate(request, cancelled=handle.cancelled, timeout=timeout)
            try:
                while True:
                    if handle.cancelled.is_set():
                        raise RequestCancelledError(f"request {handle.request_id} was cancelled")
                    handle.executing.set()
                    try:
                        event = next(iterator)
                    except StopIteration:
                        break
                    finally:
                        handle.executing.clear()
                    if event.summary is None:
                        yield chunk(event.delta if restorer is None else restorer.feed(event.delta))
                    else:
                        if restorer is not None and (remaining := restorer.finish()):
                            yield chunk(remaining)
                        self._metrics.completed(event.summary.duration_ms)
                        self._record_trace(
                            handle.request_id,
                            submitted,
                            event.summary,
                            "".join(emitted),
                            started_at,
                            armoured,
                        )
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
                                seed=event.summary.seed,
                                armoured=armoured,
                            ),
                        )
            except Exception as error:
                self._record_error(error)
                raise
            finally:
                handle.executing.clear()

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
            | RemoteProtectedMediaEngine
            | FortressDiffusionEngine,
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
            handle.executing.set()
            started_at = time.time()
            response = engine.generate_image(
                request,
                request_id=handle.request_id,
                security_mode=self.cluster.security.mode,
                cancelled=handle.cancelled,
                timeout=timeout,
            )
            self._metrics.completed(response.execution.duration_ms)
            self._record_media_trace("image", request, response, started_at)
            return response
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            handle.executing.clear()
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)
            handle.finished.set()

    def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        self._ensure_open()
        engine = self._engine(request.model)
        engine.require_modality("video")
        if not isinstance(
            engine,
            LocalDiffusionEngine
            | CompiledDiTStageEngine
            | RemoteMediaEngine
            | RemoteProtectedMediaEngine
            | FortressDiffusionEngine,
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
            handle.executing.set()
            started_at = time.time()
            response = engine.generate_video(
                request,
                request_id=handle.request_id,
                security_mode=self.cluster.security.mode,
                cancelled=handle.cancelled,
                timeout=timeout,
            )
            self._metrics.completed(response.execution.duration_ms)
            self._record_media_trace("video", request, response, started_at)
            return response
        except Exception as error:
            self._record_error(error)
            raise
        finally:
            handle.executing.clear()
            if admitted:
                self._release(request.model)
            with self._lock:
                self._requests.pop(handle.request_id, None)
            handle.finished.set()

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            handle = self._requests.get(request_id)
            if handle is None:
                return False
            handle.cancelled.set()
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed and not self._engines and self._deployment is None:
                return
            self._closed = True
            handles = list(self._requests.values())
            for handle in handles:
                handle.cancelled.set()
            executing = [handle for handle in handles if handle.executing.is_set()]
            engines = list(self._engines.values())
        deadline = time.monotonic() + self.cluster.request_timeout_seconds + 5
        for handle in executing:
            if not handle.finished.wait(max(0, deadline - time.monotonic())):
                raise RequestTimeoutError(
                    f"shutdown timed out waiting for request {handle.request_id}; "
                    "resources remain available for a close retry"
                )
        with self._lock:
            for engine in engines:
                if isinstance(engine, FortressEngine):
                    self._fortress_metrics.update(engine.transport_metrics())
            self._engines.clear()
            self._requests.clear()
        try:
            for engine in engines:
                engine.close()
        finally:
            if self._message_armour is not None:
                self._message_armour.close()
            if self._deployment is not None:
                self._deployment.close()
