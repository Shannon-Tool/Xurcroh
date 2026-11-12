from __future__ import annotations

import base64
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from safetensors.torch import load, save

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError

if TYPE_CHECKING:
    from fastapi import FastAPI

    from xurcroh.runtime import RuntimeService


def encode_public(key: X25519PublicKey) -> str:
    return base64.b64encode(key.public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()


def decode_public(value: str) -> X25519PublicKey:
    try:
        return X25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
    except ValueError as error:
        raise ProtocolError("invalid protected-media public key") from error


def seal_tensors(
    tensors: dict[str, torch.Tensor], request_id: str, session: PipelineTensorInterceptor
) -> str:
    payload = save({name: tensor.contiguous() for name, tensor in tensors.items()})
    packet = session.seal(request_id, "u8", [len(payload)], payload)
    return base64.b64encode(packet).decode()


def open_tensors(
    packet: str, request_id: str, session: PipelineTensorInterceptor
) -> dict[str, torch.Tensor]:
    try:
        encoded = base64.b64decode(packet, validate=True)
        dtype, shape, payload = session.open_tensor(encoded)
    except ValueError as error:
        raise ProtocolError("invalid protected-media envelope") from error
    if dtype != "u8" or shape != [len(payload)]:
        raise ProtocolError("protected-media envelope descriptor is invalid")
    return load(payload)


@dataclass
class MediaSession:
    session_id: str
    expires_at: float
    receiver: PipelineTensorInterceptor
    sender: PipelineTensorInterceptor


class ProtectedMediaSessionStore:
    def __init__(self, *, ttl_seconds: float = 300, capacity: int = 64) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._sessions: dict[str, MediaSession] = {}
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            self._prune()
            return len(self._sessions)

    def _prune(self) -> None:
        now = time.monotonic()
        for session_id in [key for key, value in self._sessions.items() if value.expires_at <= now]:
            self._sessions.pop(session_id, None)

    def create(self, client_public: str) -> tuple[str, str]:
        private = X25519PrivateKey.generate()
        shared = private.exchange(decode_public(client_public))
        session_id = f"media_{uuid.uuid4().hex}"
        context = session_id.encode()
        with self._lock:
            self._prune()
            if len(self._sessions) >= self.capacity:
                raise ProtocolError("protected-media session capacity is exhausted")
            self._sessions[session_id] = MediaSession(
                session_id=session_id,
                expires_at=time.monotonic() + self.ttl_seconds,
                receiver=PipelineTensorInterceptor(shared, b"worker", b"client", context),
                sender=PipelineTensorInterceptor(shared, b"worker", b"client", context),
            )
        return session_id, encode_public(private.public_key())

    def get(self, session_id: str) -> MediaSession:
        with self._lock:
            self._prune()
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise ProtocolError("unknown or expired protected-media session") from error

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        with self._lock:
            self._sessions.clear()


def install_protected_media_routes(app: FastAPI, runtime: RuntimeService) -> None:
    store = ProtectedMediaSessionStore()
    app.state.protected_media_sessions = store

    @app.post("/v1/protected/sessions")
    def create_session(values: dict[str, Any]) -> dict[str, str]:
        public = values.get("client_public")
        if not isinstance(public, str):
            raise ProtocolError("client_public is required")
        session_id, worker_public = store.create(public)
        return {"session_id": session_id, "worker_public": worker_public}

    def execute_latent(values: dict[str, Any]) -> dict[str, str]:
        session = store.get(str(values.get("session_id", "")))
        request_id = str(values.get("request_id", ""))
        model = str(values.get("model", ""))
        tensors = open_tensors(str(values.get("packet", "")), request_id, session.receiver)
        from xurcroh.runtime.diffusion import LocalDiffusionEngine
        from xurcroh.runtime.dit_stage import CompiledDiTStageEngine

        engine = runtime._engine(model)
        parameters = values.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ProtocolError("protected media parameters must be an object")
        parameters = dict(parameters)
        if isinstance(engine, CompiledDiTStageEngine):
            latent = engine.generate(
                parameters,
                latents=tensors["latents"],
                prompt_embeds=tensors["prompt_embeds"],
            )
        elif isinstance(engine, LocalDiffusionEngine):
            for name, tensor in tensors.items():
                if name not in {"latents", "basis", "permutation"}:
                    parameters[name] = tensor
            latent = engine.generate_protected_latent(
                parameters,
                latents=tensors["latents"],
                basis=tensors["basis"],
                permutation=tensors["permutation"],
            )
        else:
            raise ProtocolError("protected media execution requires a local diffusion stage")
        return {
            "session_id": session.session_id,
            "packet": seal_tensors({"latents": latent}, request_id, session.sender),
        }

    @app.post("/v1/protected/images/latents")
    def execute_image(values: dict[str, Any]) -> dict[str, str]:
        return execute_latent(values)

    @app.post("/v1/protected/videos/latents")
    def execute_video(values: dict[str, Any]) -> dict[str, str]:
        return execute_latent(values)

    @app.delete("/v1/protected/sessions/{session_id}", status_code=204)
    def close_session(session_id: str) -> None:
        store.remove(session_id)
