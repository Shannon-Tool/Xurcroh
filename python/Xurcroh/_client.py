from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, overload

from xurcroh.config import ClusterConfig, ModelRegistry, SecurityConfig
from xurcroh.errors import ConfigurationError
from xurcroh.preflight import CheckpointPreflight
from xurcroh.requests import CompletionRequest, ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime import RuntimeService
from xurcroh.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ImageGenerationResponse,
    VideoGenerationResponse,
)


class ChatCompletions:
    def __init__(self, client: Xurcroh) -> None:
        self._client = client

    @overload
    def create(self, *, stream: Literal[False] = False, **parameters: Any) -> ChatCompletion: ...

    @overload
    def create(
        self, *, stream: Literal[True], **parameters: Any
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(self, **parameters: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        request = CompletionRequest.validated(**parameters)
        if request.stream:
            return self._client._runtime.stream(request)
        return self._client._runtime.complete(request)


class Chat:
    def __init__(self, client: Xurcroh) -> None:
        self.completions = ChatCompletions(client)


class Images:
    def __init__(self, client: Xurcroh) -> None:
        self._client = client

    def generate(self, **parameters: Any) -> ImageGenerationResponse:
        return self._client._runtime.generate_image(
            ImageGenerationRequest.model_validate(parameters)
        )

    def edit(self, **parameters: Any) -> ImageGenerationResponse:
        if "image" not in parameters:
            raise ConfigurationError("image editing requires an input image")
        return self.generate(**parameters)


class Videos:
    def __init__(self, client: Xurcroh) -> None:
        self._client = client

    def generate(self, **parameters: Any) -> VideoGenerationResponse:
        return self._client._runtime.generate_video(
            VideoGenerationRequest.model_validate(parameters)
        )


class Tasks:
    """Signed traces of completed tasks, addressed by the task's request ID."""

    def __init__(self, client: Xurcroh) -> None:
        self._client = client

    def trace(self, task_id: str) -> dict[str, Any]:
        return self._client._runtime.trace(task_id)

    def proof(self, task_id: str) -> dict[str, Any]:
        return self._client._runtime.trace_proof(task_id)

    @property
    def public_key(self) -> bytes:
        return self._client._runtime.trace_public_key

    @property
    def log_key(self) -> bytes:
        return self._client._runtime.trace_log_key


class Xurcroh:
    def __init__(
        self,
        *,
        cluster: object = None,
        model_path: str | Path | Mapping[str, object],
        pipeline_parallel: int = 1,
        tensor_parallel: int = 1,
        expert_parallel: int = 1,
        security: SecurityConfig | dict[str, object] | None = None,
        provision_workers: bool | None = None,
    ) -> None:
        self._models = ModelRegistry.normalize(model_path)
        self._cluster = ClusterConfig.normalize(
            cluster,
            pipeline_parallel=pipeline_parallel,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            security=security,
            provision_workers=provision_workers,
        )
        if (
            provision_workers is None
            and not isinstance(cluster, ClusterConfig)
            and "provision_workers" not in (cluster if isinstance(cluster, Mapping) else {})
        ):
            ssh_workers = bool(self._cluster.workers) and all(
                worker.host is not None and worker.ssh is not None
                for worker in self._cluster.workers
            )
            fortress_models = any(
                definition.engine == "fortress" for _, definition in self._models.items()
            )
            if ssh_workers and fortress_models:
                self._cluster = self._cluster.model_copy(update={"provision_workers": True})
                logging.getLogger(__name__).info(
                    "Auto-enabling remote provisioning; hosts to be contacted: %s",
                    ", ".join(worker.host for worker in self._cluster.workers if worker.host),
                )
        self._runtime = RuntimeService(self._cluster, self._models)
        self.chat = Chat(self)
        self.images = Images(self)
        self.videos = Videos(self)
        self.tasks = Tasks(self)

    def cancel(self, request_id: str) -> bool:
        return self._runtime.cancel(request_id)

    def metrics(self) -> dict[str, int | float]:
        return self._runtime.metrics()

    def preflight(self, model: str | None = None) -> dict[str, CheckpointPreflight]:
        return self._runtime.preflight(model)

    def close(self) -> None:
        self._runtime.close()

    def __enter__(self) -> Xurcroh:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
