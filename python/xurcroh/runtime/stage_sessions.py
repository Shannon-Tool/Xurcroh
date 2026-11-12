from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
import uuid
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError


def encode_session_public(private: X25519PrivateKey) -> str:
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base64.b64encode(public).decode()


def decode_session_public(value: str) -> bytes:
    try:
        public = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ProtocolError("invalid stage-session public key") from error
    if len(public) != 32:
        raise ProtocolError("stage-session public key must contain 32 bytes")
    return public


def _token_digest(token: str | bytes) -> bytes:
    if isinstance(token, str):
        try:
            token = base64.b64decode(token, validate=True)
        except ValueError as error:
            raise ProtocolError("invalid stage-session token") from error
    if len(token) != 32:
        raise ProtocolError("stage-session token must contain 32 bytes")
    return hashlib.sha256(token).digest()


def _context(stage_index: int, session_id: str) -> bytes:
    return f"xurcroh-stage-{stage_index}-session-v2:{session_id}".encode()


class ClientStageSession:
    def __init__(self, stage_index: int, recipient_public: bytes) -> None:
        if len(recipient_public) != 32:
            raise ValueError("recipient public key must contain 32 bytes")
        self.stage_index = stage_index
        self.session_id = uuid.uuid4().hex
        self._token = secrets.token_bytes(32)
        self.session_token = base64.b64encode(self._token).decode()
        self._private = X25519PrivateKey.generate()
        self.client_public = encode_session_public(self._private)
        shared = self._private.exchange(X25519PublicKey.from_public_bytes(recipient_public))
        context = _context(stage_index, self.session_id)
        self.sender = PipelineTensorInterceptor(shared, b"client", b"stage", context)
        self.receiver = PipelineTensorInterceptor(shared, b"client", b"stage", context)

    def auth_headers(self) -> dict[str, str]:
        return {
            "X-Xurcroh-Session-Id": self.session_id,
            "X-Xurcroh-Session-Token": self.session_token,
        }

    @property
    def token_bytes(self) -> bytes:
        return self._token

    def wire_packet(self, packet: bytes) -> bytes:
        return (
            decode_session_public(self.client_public)
            + self.session_id.encode()
            + self._token
            + packet
        )


@dataclass
class _ServerSession:
    expires_at: float
    client_public: bytes
    token_digest: bytes
    receiver: PipelineTensorInterceptor
    sender: PipelineTensorInterceptor


class StageSessionStore:
    def __init__(
        self,
        private_key: X25519PrivateKey,
        stage_index: int,
        *,
        ttl_seconds: float = 300,
        capacity: int = 64,
    ) -> None:
        self._private = private_key
        self.stage_index = stage_index
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._sessions: dict[str, _ServerSession] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.monotonic()
        for session_id in [key for key, value in self._sessions.items() if value.expires_at <= now]:
            self._sessions.pop(session_id, None)

    def get_or_create(
        self, session_id: str, client_public: str, session_token: str | bytes
    ) -> tuple[PipelineTensorInterceptor, PipelineTensorInterceptor]:
        if len(session_id) != 32:
            raise ProtocolError("invalid stage-session identifier")
        try:
            uuid.UUID(session_id)
        except ValueError as error:
            raise ProtocolError("invalid stage-session identifier") from error
        public = decode_session_public(client_public)
        token_digest = _token_digest(session_token)
        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.client_public != public or not hmac.compare_digest(
                    existing.token_digest, token_digest
                ):
                    raise ProtocolError("stage-session credentials do not match")
                existing.expires_at = time.monotonic() + self.ttl_seconds
                return existing.receiver, existing.sender
            if len(self._sessions) >= self.capacity:
                raise ProtocolError("stage-session capacity is exhausted")
            shared = self._private.exchange(X25519PublicKey.from_public_bytes(public))
            context = _context(self.stage_index, session_id)
            session = _ServerSession(
                expires_at=time.monotonic() + self.ttl_seconds,
                client_public=public,
                token_digest=token_digest,
                receiver=PipelineTensorInterceptor(shared, b"stage", b"client", context),
                sender=PipelineTensorInterceptor(shared, b"stage", b"client", context),
            )
            self._sessions[session_id] = session
            return session.receiver, session.sender

    def authorize(self, session_id: str, session_token: str | bytes) -> None:
        token_digest = _token_digest(session_token)
        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is None or not hmac.compare_digest(existing.token_digest, token_digest):
                raise ProtocolError("stage-session credentials are not authorized")

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        with self._lock:
            self._sessions.clear()

    @property
    def active_count(self) -> int:
        with self._lock:
            self._prune()
            return len(self._sessions)
