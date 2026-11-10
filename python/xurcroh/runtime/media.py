from __future__ import annotations

import threading
import time
from typing import Any, Literal

import httpx

from xurcroh.config import SecurityConfig, WorkerConfig
from xurcroh.errors import ClusterUnavailableError
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.generation import check_request_health
from xurcroh.runtime.remote import RemoteOpenAIEngine
from xurcroh.types import ImageGenerationResponse, VideoGenerationResponse


class RemoteMediaEngine:
    def __init__(
        self,
        workers: list[WorkerConfig],
        security: SecurityConfig,
        modality: Literal["image", "video"],
    ) -> None:
        self.workers = workers
        self.modality = modality
        self.engine_name = "remote-diffusers"
        self.worker_id = workers[0].name
        self.worker_ids = [worker.name for worker in workers]
        self._client = httpx.Client(timeout=60)
        self._closed = False
        for worker in workers:
            RemoteOpenAIEngine.validate_worker(worker, security)

    def require_modality(self, expected: str) -> None:
        if expected != self.modality:
            raise ClusterUnavailableError(
                f"remote {self.modality} engine does not support {expected} generation"
            )

    @staticmethod
    def _endpoint(worker: WorkerConfig) -> str:
        endpoint = worker.endpoint or ""
        if endpoint.startswith("http+dev://"):
            return "http://" + endpoint.removeprefix("http+dev://")
        return endpoint.rstrip("/")

    @staticmethod
    def _headers(worker: WorkerConfig) -> dict[str, str]:
        return RemoteOpenAIEngine._headers(worker)

    def _post(
        self,
        path: str,
        values: dict[str, Any],
        *,
        cancelled: threading.Event,
        started: float,
        timeout: float,
    ) -> dict[str, Any]:
        errors = []
        for worker in self.workers:
            try:
                check_request_health(cancelled, started, timeout)
                response = self._client.post(
                    f"{self._endpoint(worker)}{path}",
                    json=values,
                    headers=self._headers(worker),
                    timeout=max(timeout - (time.monotonic() - started), 0.001),
                )
                response.raise_for_status()
                result = response.json()
                check_request_health(cancelled, started, timeout)
                if not isinstance(result, dict):
                    raise ValueError("media response must be an object")
                return result
            except (httpx.HTTPError, ValueError) as error:
                errors.append(f"{worker.name}: {type(error).__name__}")
        raise ClusterUnavailableError("all remote media workers failed: " + ", ".join(errors))

    def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: threading.Event,
        timeout: float,
    ) -> ImageGenerationResponse:
        started = time.monotonic()
        result = self._post(
            "/v1/images/generations",
            request.model_dump(mode="json", exclude_none=True),
            cancelled=cancelled,
            started=started,
            timeout=timeout,
        )
        response = ImageGenerationResponse.model_validate(result)
        execution = response.execution.model_copy(
            update={
                "request_id": request_id,
                "engine": self.engine_name,
                "worker_ids": self.worker_ids,
                "security_mode": security_mode,
            }
        )
        return response.model_copy(update={"execution": execution})

    def generate_video(
        self,
        request: VideoGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: threading.Event,
        timeout: float,
    ) -> VideoGenerationResponse:
        started = time.monotonic()
        result = self._post(
            "/v1/videos/generations",
            request.model_dump(mode="json", exclude_none=True),
            cancelled=cancelled,
            started=started,
            timeout=timeout,
        )
        response = VideoGenerationResponse.model_validate(result)
        execution = response.execution.model_copy(
            update={
                "request_id": request_id,
                "engine": self.engine_name,
                "worker_ids": self.worker_ids,
                "security_mode": security_mode,
            }
        )
        return response.model_copy(update={"execution": execution})

    def health(self) -> dict[str, Any]:
        response = self._client.get(f"{self._endpoint(self.workers[0])}/health")
        response.raise_for_status()
        return dict(response.json())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()
