from __future__ import annotations

import base64
import contextlib
import threading
import time
from typing import Any

import httpx
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
from xurcroh.errors import ClusterUnavailableError, ConfigurationError
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.attestation import RemoteAttestationClient
from xurcroh.runtime.generation import check_request_health
from xurcroh.runtime.media_boundary import DiffusionMediaBoundary
from xurcroh.runtime.media_sessions import decode_public, encode_public, open_tensors, seal_tensors
from xurcroh.runtime.remote import RemoteOpenAIEngine
from xurcroh.types import (
    ExecutionMetadata,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResponse,
    VideoGenerationResponse,
)


class RemoteProtectedMediaEngine:
    def __init__(
        self,
        name: str,
        definition: ModelDefinition,
        workers: list[WorkerConfig],
        security: SecurityConfig,
    ) -> None:
        self.name = name
        self.definition = definition
        self.workers = workers
        self.engine_name = "protected-remote-diffusers"
        self.worker_id = workers[0].name
        self.worker_ids = [worker.name for worker in workers]
        self.modality = definition.modality
        self._boundary = DiffusionMediaBoundary.from_definition(definition)
        self._client = httpx.Client(timeout=60)
        self._closed = False
        for worker in workers:
            RemoteOpenAIEngine.validate_worker(worker, security, stage=True)
            if security.require_attestation:
                RemoteAttestationClient(self._client).verify_worker(worker.endpoint or "")

    def require_modality(self, expected: str) -> None:
        if expected != self.modality:
            raise ConfigurationError(
                f"protected {self.modality} engine does not support {expected}"
            )

    @staticmethod
    def _endpoint(worker: WorkerConfig) -> str:
        endpoint = worker.endpoint or ""
        if endpoint.startswith("http+dev://"):
            return "http://" + endpoint.removeprefix("http+dev://")
        return endpoint.rstrip("/")

    def _metadata(self, request_id: str, started: float, security_mode: str) -> ExecutionMetadata:
        return ExecutionMetadata(
            request_id=request_id,
            engine=self.engine_name,
            worker_ids=self.worker_ids,
            duration_ms=max((time.monotonic() - started) * 1000, 0.001),
            security_mode=security_mode,
        )

    def _execute(
        self,
        path: str,
        request_id: str,
        model: str,
        tensors: dict[str, torch.Tensor],
        parameters: dict[str, Any],
        *,
        cancelled: threading.Event,
        started: float,
        timeout: float,
    ) -> torch.Tensor:
        private = X25519PrivateKey.generate()
        worker = self.workers[0]
        session_id: str | None = None
        try:
            handshake = self._client.post(
                f"{self._endpoint(worker)}/v1/protected/sessions",
                json={"client_public": encode_public(private.public_key())},
                timeout=timeout,
            )
            handshake.raise_for_status()
            session = handshake.json()
            session_id = session["session_id"]
            shared = private.exchange(decode_public(session["worker_public"]))
            context = session_id.encode()
            sender = PipelineTensorInterceptor(shared, b"client", b"worker", context)
            receiver = PipelineTensorInterceptor(shared, b"worker", b"client", context)
            response = self._client.post(
                f"{self._endpoint(worker)}{path}",
                json={
                    "session_id": session_id,
                    "request_id": request_id,
                    "model": model,
                    "packet": seal_tensors(tensors, request_id, sender),
                    "parameters": parameters,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            result = open_tensors(response.json()["packet"], request_id, receiver)
            check_request_health(cancelled, started, timeout)
            return result["latents"]
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise ClusterUnavailableError(
                f"protected media worker {worker.name} failed: {type(error).__name__}"
            ) from error
        finally:
            if session_id is not None:
                with contextlib.suppress(httpx.HTTPError):
                    self._client.delete(
                        f"{self._endpoint(worker)}/v1/protected/sessions/{session_id}",
                        timeout=10,
                    )

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
        check_request_health(cancelled, started, timeout)
        tensors, parameters = self._boundary.prepare(request)
        latent = self._execute(
            "/v1/protected/images/latents",
            request_id,
            request.model,
            tensors,
            parameters,
            cancelled=cancelled,
            started=started,
            timeout=timeout,
        )
        encoded = self._boundary.decode(latent, request)
        check_request_health(cancelled, started, timeout)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[GeneratedImage(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode),
        )

    def generate_video(
        self,
        request: VideoGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: threading.Event,
        timeout: float,
    ) -> VideoGenerationResponse:
        if request.image is not None:
            raise ConfigurationError(
                "protected image-to-video requires client-side source-latent encoding"
            )
        started = time.monotonic()
        check_request_health(cancelled, started, timeout)
        tensors, parameters = self._boundary.prepare_video(request)
        latent = self._execute(
            "/v1/protected/videos/latents",
            request_id,
            request.model,
            tensors,
            parameters,
            cancelled=cancelled,
            started=started,
            timeout=timeout,
        )
        encoded = self._boundary.decode_video(latent, request)
        check_request_health(cancelled, started, timeout)
        return VideoGenerationResponse(
            created=int(time.time()),
            data=[GeneratedVideo(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()
