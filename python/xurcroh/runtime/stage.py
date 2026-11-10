from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import torch

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.client.boundary import ClientBoundary
from xurcroh.config import ModelDefinition, SecurityConfig, WorkerConfig
from xurcroh.errors import (
    ClusterUnavailableError,
    ConfigurationError,
    ModelLoadError,
    ProtocolError,
    RequestTimeoutError,
    ShardIntegrityError,
)
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.attestation import RemoteAttestationClient
from xurcroh.runtime.generation import (
    GenerationEvent,
    GenerationSummary,
    check_request_health,
    stopping_strings,
)
from xurcroh.runtime.remote import RemoteOpenAIEngine
from xurcroh.runtime.stage_sessions import ClientStageSession
from xurcroh.shards.recipients import verify_manifest_digest


class DenseStageEngine:
    def __init__(
        self,
        name: str,
        definition: ModelDefinition,
        workers: list[WorkerConfig],
        security: SecurityConfig,
        controller_endpoint: str | None = None,
    ) -> None:
        if definition.manifest_path is None:
            raise ConfigurationError("dense stage execution requires manifest_path")
        self.name = name
        self.definition = definition
        self.workers = workers
        self.engine_name = "dense-stages"
        self.worker_id = workers[0].name
        self.worker_ids = [worker.name for worker in workers]
        self.modality = "text"
        self.controller_endpoint = controller_endpoint
        self._closed = False
        manifest_path = Path(definition.manifest_path)
        try:
            document = json.loads(manifest_path.read_text())
            if document["format"] != "xurcroh-manifest" or document["version"] != 2:
                raise ShardIntegrityError("unsupported dense stage manifest")
            verify_manifest_digest(document)
            if document["stage_count"] != len(workers):
                raise ConfigurationError("pipeline worker count must equal manifest stage_count")
            stages = document["stages"]
            boundary = document["boundary"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ShardIntegrityError("invalid dense stage manifest") from error
        environment_name = definition.boundary_key_env or "XURCROH_CLIENT_BOUNDARY_KEY"
        encoded_key = os.environ.get(environment_name)
        if encoded_key is None:
            raise ConfigurationError(
                f"client boundary key environment variable {environment_name} is not set"
            )
        try:
            boundary_key = bytes.fromhex(encoded_key)
        except ValueError as error:
            raise ConfigurationError("client boundary key must be hexadecimal") from error
        if len(boundary_key) != 32:
            raise ConfigurationError("client boundary key must contain 32 bytes")
        self._client = httpx.Client(timeout=60)
        self._native_clients: dict[int, Any] = {}
        self._native_locks = [threading.Lock() for _ in workers]
        self._recipient_public: list[bytes] = []
        for stage in stages:
            if Path(stage["file"]).name != stage["file"]:
                raise ShardIntegrityError("unsafe stage artifact path")
            if Path(stage["key_envelope"]).name != stage["key_envelope"]:
                raise ShardIntegrityError("unsafe stage key path")
            recipient_public = bytes.fromhex(stage["recipient_public"])
            if len(recipient_public) != 32:
                raise ShardIntegrityError("invalid stage recipient public key")
            self._recipient_public.append(recipient_public)
        if Path(boundary["file"]).name != boundary["file"]:
            raise ShardIntegrityError("unsafe client boundary path")
        self.boundary = ClientBoundary(manifest_path.parent / boundary["file"], boundary_key)
        for worker in workers:
            RemoteOpenAIEngine.validate_worker(worker, security, stage=True)
            if security.require_attestation:
                RemoteAttestationClient(self._client).verify_worker(
                    worker.labels.get("attestation_endpoint", worker.endpoint or "")
                )
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                definition.path,
                local_files_only=not definition.allow_download,
                trust_remote_code=False,
                revision=definition.revision,
            )
            config = json.loads((Path(definition.path) / "config.json").read_text())
            text_config = {**config, **config.get("text_config", {})}
            self.query_heads = int(text_config["num_attention_heads"])
            self.kv_heads = int(text_config["num_key_value_heads"])
            self.position_limit = int(text_config["max_position_embeddings"])
            self.vocabulary_size = int(text_config["vocab_size"])
            self.interleaved_rope = text_config.get("rope_style") == "interleaved"
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            self.boundary.close()
            self._client.close()
            raise ModelLoadError(
                f"cannot load dense client tokenizer/configuration: {error}"
            ) from error

    def require_modality(self, expected: str) -> None:
        if expected != "text":
            raise ConfigurationError("dense stage engine only supports text completions")

    @staticmethod
    def _endpoint(worker: WorkerConfig) -> str:
        endpoint = worker.endpoint or ""
        if endpoint.startswith("http+dev://"):
            return "http://" + endpoint.removeprefix("http+dev://")
        return endpoint.rstrip("/")

    def _controller_url(self, path: str) -> str:
        endpoint = self.controller_endpoint or ""
        if endpoint.startswith("http+dev://"):
            endpoint = "http://" + endpoint.removeprefix("http+dev://")
        return f"{endpoint.rstrip('/')}{path}"

    def _create_circuit(self, model: str) -> str:
        try:
            response = self._client.post(
                self._controller_url("/v1/circuits"),
                json={
                    "model": model,
                    "pipeline_parallel": len(self.workers),
                    "tensor_parallel": 1,
                    "expert_parallel": 1,
                },
            )
            response.raise_for_status()
            return str(response.json()["circuit_id"])
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise ClusterUnavailableError("failed to create controller circuit") from error

    def _delete_circuit(self, circuit_id: str) -> None:
        with suppress(httpx.HTTPError):
            self._client.delete(self._controller_url(f"/v1/circuits/{circuit_id}"), timeout=10)

    def _native_client(self, index: int) -> Any:
        client = self._native_clients.get(index)
        if client is not None:
            return client
        import xurcroh_quic

        worker = self.workers[index]
        ca_path = worker.labels.get("ca_path")
        if ca_path is None:
            raise ConfigurationError(f"QUIC worker {worker.name} requires labels.ca_path")
        certificate = Path(ca_path).read_bytes()
        client = xurcroh_quic.TensorClient(
            (worker.endpoint or "").removeprefix("quic://"),
            worker.labels.get("server_name", worker.name),
            certificate,
        )
        self._native_clients[index] = client
        return client

    def _execute_native(
        self,
        index: int,
        packet: bytes,
        request_id: str,
        shape: list[int],
        receiver: PipelineTensorInterceptor,
        timeout: float,
    ) -> torch.Tensor:
        client = self._native_client(index)
        request_hash = hashlib.sha256(request_id.encode()).digest()[:16]

        def round_trip() -> Any:
            client.send(request_hash, index, 0, "tensor", "u8", [len(packet)], packet)
            return client.receive()

        try:
            with self._native_locks[index], ThreadPoolExecutor(max_workers=1) as executor:
                response = executor.submit(round_trip).result(timeout=max(timeout, 0.001))
        except FutureTimeoutError as error:
            with suppress(ValueError):
                client.close()
            self._native_clients.pop(index, None)
            raise RequestTimeoutError(f"native stage {index} timed out") from error
        except ValueError as error:
            self._native_clients.pop(index, None)
            raise ClusterUnavailableError(
                f"native stage {index} failed: {type(error).__name__}"
            ) from error
        response_id, _, _, kind, dtype, response_shape, result = response
        if response_id != request_hash or kind != "tensor" or dtype != "u8":
            raise ProtocolError("native stage response metadata is invalid")
        if response_shape != [len(result)]:
            raise ProtocolError("native stage response length is invalid")
        result_dtype, result_shape, payload = receiver.open_tensor(result)
        if (
            result_dtype != "f64"
            or result_shape != shape
            or len(payload) != shape[0] * shape[1] * 8
        ):
            raise ProtocolError("native stage response tensor does not match the request shape")
        buffer = bytearray(payload)
        try:
            return torch.frombuffer(buffer, dtype=torch.float64).reshape(shape).clone()
        finally:
            buffer[:] = bytes(len(buffer))

    def _execute_stage(
        self,
        index: int,
        hidden: torch.Tensor,
        request_id: str,
        session: ClientStageSession,
        timeout: float,
        circuit_id: str | None = None,
    ) -> torch.Tensor:
        contiguous = hidden.detach().cpu().contiguous()
        if contiguous.dtype != torch.float64 or contiguous.ndim != 2:
            raise ProtocolError("client stage input must be a float64 matrix")
        shape = list(contiguous.shape)
        packet = session.sender.seal(request_id, "f64", shape, contiguous.numpy().tobytes())
        worker = self.workers[index]
        if (worker.endpoint or "").startswith("quic://"):
            wire_packet = session.wire_packet(packet)
            return self._execute_native(
                index, wire_packet, request_id, shape, session.receiver, timeout
            )
        url = (
            self._controller_url(f"/v1/relay/{circuit_id}")
            if circuit_id is not None
            else f"{self._endpoint(worker)}/v1/stages/execute"
        )
        headers = {"X-Xurcroh-Stage": str(index)} if circuit_id is not None else {}
        try:
            response = self._client.post(
                url,
                json={
                    "request_id": request_id,
                    "session_id": session.session_id,
                    "client_public": session.client_public,
                    "stage_index": index,
                    "dtype": "f64",
                    "shape": shape,
                    "packet": base64.b64encode(packet).decode(),
                },
                headers=headers,
                timeout=max(timeout, 0.001),
            )
            response.raise_for_status()
            values = response.json()
            if values["stage_index"] != index or values["dtype"] != "f64":
                raise ProtocolError("stage response metadata is invalid")
            result = base64.b64decode(values["packet"], validate=True)
        except (httpx.HTTPError, ValueError, KeyError, binascii.Error) as error:
            raise ClusterUnavailableError(
                f"stage {index} worker {worker.name} failed: {type(error).__name__}"
            ) from error
        dtype, result_shape, payload = session.receiver.open_tensor(result)
        if dtype != "f64" or result_shape != shape or len(payload) != shape[0] * shape[1] * 8:
            raise ProtocolError("stage response tensor does not match the request shape")
        buffer = bytearray(payload)
        try:
            return torch.frombuffer(buffer, dtype=torch.float64).reshape(shape).clone()
        finally:
            buffer[:] = bytes(len(buffer))

    def _clear_stage_caches(
        self,
        request_id: str,
        circuit_id: str | None,
        sessions: list[ClientStageSession],
    ) -> None:
        if circuit_id is not None:
            with suppress(httpx.HTTPError):
                self._client.delete(
                    self._controller_url(f"/v1/relay/{circuit_id}/stages/{request_id}"),
                    timeout=10,
                )
        for index, (worker, session) in enumerate(zip(self.workers, sessions, strict=False)):
            if (worker.endpoint or "").startswith("quic://"):
                client = self._native_clients.get(index)
                if client is not None:
                    session_uuid = uuid.UUID(session.session_id).bytes

                    def cancel_native(
                        client: Any = client,
                        session_uuid: bytes = session_uuid,
                        index: int = index,
                    ) -> None:
                        client.send(
                            session_uuid,
                            index,
                            0,
                            "cancel",
                            "",
                            [],
                            b"",
                        )
                        response = client.receive()
                        if response[3] != "complete":
                            raise ValueError("native stage cancellation was not acknowledged")

                    with (
                        suppress(ValueError, FutureTimeoutError),
                        self._native_locks[index],
                        ThreadPoolExecutor(max_workers=1) as executor,
                    ):
                        executor.submit(cancel_native).result(timeout=10)
            else:
                with suppress(httpx.HTTPError):
                    self._client.delete(
                        f"{self._endpoint(worker)}/v1/stages/{request_id}", timeout=10
                    )
                with suppress(httpx.HTTPError):
                    self._client.delete(
                        f"{self._endpoint(worker)}/v1/stages/sessions/{session.session_id}",
                        timeout=10,
                    )

    def generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        circuit_id = (
            self._create_circuit(request.model) if self.controller_endpoint is not None else None
        )
        request_id = request.request_id or f"req_{uuid.uuid4().hex}"
        sessions = [
            ClientStageSession(index, recipient_public)
            for index, recipient_public in enumerate(self._recipient_public)
        ]
        try:
            yield from self._generate(
                request,
                cancelled=cancelled,
                timeout=timeout,
                circuit_id=circuit_id,
                request_id=request_id,
                sessions=sessions,
            )
        finally:
            self._clear_stage_caches(request_id, circuit_id, sessions)
            if circuit_id is not None:
                self._delete_circuit(circuit_id)

    def _generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
        circuit_id: str | None,
        request_id: str,
        sessions: list[ClientStageSession],
    ) -> Iterator[GenerationEvent]:
        if self._closed:
            raise ModelLoadError("dense stage engine is closed")
        started = time.monotonic()
        prompt = "\n".join(
            f"{message.role}: {message.plain_text()}" for message in request.messages
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        prompt_tokens = token_ids.shape[-1]
        if prompt_tokens < 1 or prompt_tokens + request.max_tokens > self.position_limit:
            raise ConfigurationError("prompt and max_tokens exceed the model context limit")
        if int(token_ids.max()) >= self.vocabulary_size:
            raise ConfigurationError("tokenizer emits IDs outside the model vocabulary")
        completion_ids: list[int] = []
        text = ""
        finish_reason: Literal["stop", "length"] = "length"
        stops = stopping_strings(request)
        for token_index in range(request.max_tokens):
            check_request_health(cancelled, started, timeout)
            boundary_input = token_ids if token_index == 0 else token_ids[:, -1:]
            hidden = self.boundary.embed(boundary_input)[0]
            for index in range(len(self.workers)):
                check_request_health(cancelled, started, timeout)
                remaining = timeout - (time.monotonic() - started)
                hidden = self._execute_stage(
                    index, hidden, request_id, sessions[index], remaining, circuit_id
                )
                if index + 1 < len(self.workers):
                    hidden = self.boundary.transition(hidden, index)
                check_request_health(cancelled, started, timeout)
            next_token = self.boundary.sample(hidden[-1:], request)
            token_id = int(next_token.item())
            completion_ids.append(token_id)
            updated = cast(str, self._tokenizer.decode(completion_ids, skip_special_tokens=True))
            delta = updated[len(text) :]
            text = updated
            token_ids = torch.cat([token_ids, next_token.reshape(1, 1)], dim=1)
            hidden.zero_()
            yield GenerationEvent(delta=delta)
            if token_id == self._tokenizer.eos_token_id:
                finish_reason = "stop"
                break
            if stops and any(text.endswith(stop) for stop in stops):
                finish_reason = "stop"
                break
        duration_ms = (time.monotonic() - started) * 1000
        yield GenerationEvent(
            delta="",
            summary=GenerationSummary(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(completion_ids),
                finish_reason=finish_reason,
                duration_ms=duration_ms,
            ),
        )

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "stages": len(self.workers)}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for client in self._native_clients.values():
                with suppress(ValueError):
                    client.close()
            self._native_clients.clear()
            self.boundary.close()
            self._client.close()
