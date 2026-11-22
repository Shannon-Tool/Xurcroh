from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
import uuid
from dataclasses import dataclass

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError
from xurcroh.kem import (
    PUBLIC_KEY_BYTES,
    SEAL_OVERHEAD_BYTES,
    HybridKemKey,
    KemError,
    open_sealed,
    seal,
)

# The session master sealed to the stage's key with ML-KEM-768 and X25519 together.
HPKE_CIPHERTEXT_BYTES = SEAL_OVERHEAD_BYTES + 32
SESSION_ENVELOPE_BYTES = 32 + 32 + 32 + HPKE_CIPHERTEXT_BYTES


def _handshake_digest(handshake: bytes) -> bytes:
    return hashlib.sha256(b"xurcroh-stage-handshake-v3" + handshake).digest()


def decode_session_public(value: str) -> bytes:
    """Decode the 32-byte digest that names a session's sealed handshake."""

    try:
        public = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ProtocolError("invalid stage-session handshake digest") from error
    if len(public) != 32:
        raise ProtocolError("stage-session handshake digest must contain 32 bytes")
    return public


def _decode_token(token: str | bytes) -> bytes:
    if isinstance(token, str):
        try:
            token = base64.b64decode(token, validate=True)
        except ValueError as error:
            raise ProtocolError("invalid stage-session token") from error
    if len(token) != 32:
        raise ProtocolError("stage-session token must contain 32 bytes")
    return token


def _token_digest(token: str | bytes) -> bytes:
    return hashlib.sha256(_decode_token(token)).digest()


def _context(stage_index: int, session_id: str) -> bytes:
    return f"xurcroh-stage-{stage_index}-session-v2:{session_id}".encode()


def _hpke_info(stage_index: int, session_id: str, token: bytes) -> bytes:
    # Binds the sealed master to the stage, the session and its bearer token.
    return f"xurcroh-stage-hpke-v3:{stage_index}:{session_id}:".encode() + token


class ClientStageSession:
    def __init__(self, stage_index: int, recipient_public: bytes) -> None:
        if len(recipient_public) != PUBLIC_KEY_BYTES:
            raise ValueError(f"recipient public key must contain {PUBLIC_KEY_BYTES} bytes")
        self.stage_index = stage_index
        self.session_id = uuid.uuid4().hex
        self._token = secrets.token_bytes(32)
        self.session_token = base64.b64encode(self._token).decode()
        master = secrets.token_bytes(32)
        self.handshake: bytes = seal(
            recipient_public, _hpke_info(stage_index, self.session_id, self._token), master
        )
        self.client_public = base64.b64encode(_handshake_digest(self.handshake)).decode()
        context = _context(stage_index, self.session_id)
        self.sender = PipelineTensorInterceptor(master, b"client", b"stage", context)
        self.receiver = PipelineTensorInterceptor(master, b"client", b"stage", context)

    def auth_headers(self) -> dict[str, str]:
        return {
            "X-Xurcroh-Session-Id": self.session_id,
            "X-Xurcroh-Session-Token": self.session_token,
        }

    @property
    def token_bytes(self) -> bytes:
        return self._token

    def rotate(self, epoch: int) -> None:
        if epoch <= 0:
            raise ProtocolError("stage-session epoch must increase")
        self.sender.rotate(epoch)
        self.receiver.rotate(epoch)

    def wire_packet(self, packet: bytes) -> bytes:
        return (
            decode_session_public(self.client_public)
            + self.session_id.encode()
            + self._token
            + self.handshake
            + packet
        )


@dataclass
class _ServerSession:
    expires_at: float
    client_public: bytes
    token_digest: bytes
    receiver: PipelineTensorInterceptor
    sender: PipelineTensorInterceptor
    epoch: int = 0


class StageSessionStore:
    def __init__(
        self,
        private_key: HybridKemKey,
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
        self,
        session_id: str,
        client_public: str,
        session_token: str | bytes,
        handshake: bytes,
    ) -> tuple[PipelineTensorInterceptor, PipelineTensorInterceptor]:
        if len(session_id) != 32:
            raise ProtocolError("invalid stage-session identifier")
        try:
            uuid.UUID(session_id)
        except ValueError as error:
            raise ProtocolError("invalid stage-session identifier") from error
        public = decode_session_public(client_public)
        token = _decode_token(session_token)
        token_digest = _token_digest(token)
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
            if len(handshake) != HPKE_CIPHERTEXT_BYTES:
                raise ProtocolError("stage-session HPKE ciphertext is invalid")
            if not hmac.compare_digest(public, _handshake_digest(handshake)):
                raise ProtocolError("stage-session handshake digest does not match")
            try:
                master = open_sealed(
                    self._private, _hpke_info(self.stage_index, session_id, token), handshake
                )
            except KemError as error:
                raise ProtocolError("stage-session HPKE ciphertext does not open") from error
            if len(master) != 32:
                raise ProtocolError("stage-session master must contain 32 bytes")
            context = _context(self.stage_index, session_id)
            session = _ServerSession(
                expires_at=time.monotonic() + self.ttl_seconds,
                client_public=public,
                token_digest=token_digest,
                receiver=PipelineTensorInterceptor(master, b"stage", b"client", context),
                sender=PipelineTensorInterceptor(master, b"stage", b"client", context),
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

    def rotate(self, session_id: str, epoch: int) -> None:
        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is None or epoch != existing.epoch + 1:
                raise ProtocolError("stage-session rotation epoch is invalid")
            existing.receiver.rotate(epoch)
            existing.sender.rotate(epoch)
            existing.epoch = epoch

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
