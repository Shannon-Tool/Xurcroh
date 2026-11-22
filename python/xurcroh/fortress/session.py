"""Forward-secret, post-quantum session establishment for Fortress role workers.

One round trip before the first call:

1. The client sends a fresh ephemeral hybrid public key and a secret sealed to
   the worker's static recipient key.
2. The worker opens that secret, seals a secret of its own to the client's
   ephemeral key, and returns it with the digest of the artifacts it loaded and,
   when it has an identity key, a signature over the whole exchange.

The session keys are derived from both secrets and a hash of the exchange. The
worker's static key alone therefore cannot open a recorded session: that also
needs the client's ephemeral key, which the client discards after the handshake.
Both key exchanges are X25519 with ML-KEM-768. Keys are then replaced by a
one-way ratchet at every rotation, so a key taken from memory does not open
earlier traffic.

Python cannot guarantee that discarded key bytes are erased from process memory;
erasure here is best effort.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from xurcroh import kem
from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError
from xurcroh.signatures import (
    SIGNATURE_BYTES,
    HybridSigningKey,
    SignatureError,
    verify_hybrid,
)

HANDSHAKE_MAGIC = b"XFH1"
DATA_MAGIC = b"XFD1"
SESSION_ID_BYTES = 32
BINDING_BYTES = 32
SECRET_BYTES = 32
ARTIFACT_DIGEST_BYTES = 32
HANDSHAKE_CONTEXT = b"xurcroh-role-handshake"
SESSION_SUITE = f"{kem.SUITE_NAME}; ChaCha20-Poly1305 frames; one-way key ratchet"
_DOMAIN = b"xurcroh-role-session-v3"
_SEALED_SECRET_BYTES = kem.SEAL_OVERHEAD_BYTES + SECRET_BYTES
_HELLO_BYTES = len(HANDSHAKE_MAGIC) + SESSION_ID_BYTES + kem.PUBLIC_KEY_BYTES + _SEALED_SECRET_BYTES
_REPLY_BYTES = _SEALED_SECRET_BYTES + ARTIFACT_DIGEST_BYTES
# Used when the client's TLS session ends at a relay, so there is no TLS session
# shared with the worker to bind to.
UNBOUND = bytes(BINDING_BYTES)


def channel_binding(link: Any, *, relayed: bool) -> bytes:
    """A value both ends of one TLS connection derive and nobody else can.

    Mixing it into the handshake ties the session to that TLS connection. On a
    relayed route there is no such shared connection and the session is unbound.
    """

    if relayed:
        return UNBOUND
    return bytes(
        link.export_keying_material(b"EXPORTER-xurcroh-role-session-v3", b"", BINDING_BYTES)
    )


def _hash(*parts: bytes) -> bytes:
    digest = hashlib.sha256(_DOMAIN)
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def _hkdf(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(key)


def _session_id(value: bytes) -> str:
    try:
        session_id = value.decode("ascii")
        uuid.UUID(hex=session_id)
    except (UnicodeDecodeError, ValueError) as error:
        raise ProtocolError("invalid role-session identifier") from error
    if len(session_id) != SESSION_ID_BYTES:
        raise ProtocolError("invalid role-session identifier")
    return session_id


class RatchetedChannel(PipelineTensorInterceptor):
    """An encrypted tensor channel whose key moves forward one way at each rotation."""

    def __init__(
        self, master_key: bytes, send_label: bytes, receive_label: bytes, context: bytes
    ) -> None:
        super().__init__(master_key, send_label, receive_label, context)
        # The base class keeps the first key to re-derive from; a ratchet must not.
        self._master_key = b""
        self._chain = bytearray(master_key)

    def rotate(self, epoch: int) -> None:
        following = bytearray(
            _hkdf(bytes(self._chain), b"xurcroh-role-ratchet-v1", epoch.to_bytes(8, "big"), 32)
        )
        self._session.rotate(
            bytes(following), self._send_label, self._receive_label, self._context, epoch
        )
        for index in range(len(self._chain)):
            self._chain[index] = 0
        self._chain = following


@dataclass(frozen=True)
class _Keys:
    master: bytes
    token: bytes


def _derive(client_secret: bytes, worker_secret: bytes, transcript: bytes) -> _Keys:
    material = _hkdf(client_secret + worker_secret, _DOMAIN, transcript, 64)
    return _Keys(master=material[:32], token=material[32:])


def _context(weight_index: int, session_id: str) -> bytes:
    return f"xurcroh-role-{weight_index}-session-v3:{session_id}".encode()


class ClientRoleSession:
    """The client's half of one role session."""

    def __init__(
        self,
        weight_index: int,
        recipient_public: bytes,
        *,
        binding: bytes = UNBOUND,
        identity_public: bytes | None = None,
        artifact_digest: str | None = None,
    ) -> None:
        if len(binding) != BINDING_BYTES:
            raise ValueError("channel binding must contain 32 bytes")
        self.weight_index = weight_index
        self.session_id = uuid.uuid4().hex
        self.established = False
        self.worker_signed = False
        self.tls_bound = binding != UNBOUND
        self.artifact_digest: str | None = None
        self._identity_public = identity_public
        self._expected_artifacts = artifact_digest
        self._ephemeral: kem.HybridKemKey | None = kem.HybridKemKey.generate()
        self._secret = secrets.token_bytes(SECRET_BYTES)
        index = weight_index.to_bytes(4, "big")
        self._opening = _hash(
            index, self.session_id.encode(), binding, recipient_public, self._ephemeral.public_key
        )
        self._sealed = kem.seal(recipient_public, b"client:" + self._opening, self._secret)
        self._token = b""
        self.sender: RatchetedChannel
        self.receiver: RatchetedChannel

    def hello(self) -> bytes:
        if self._ephemeral is None:
            raise ProtocolError("role-session handshake is already complete")
        return (
            HANDSHAKE_MAGIC + self.session_id.encode() + self._ephemeral.public_key + self._sealed
        )

    def finish(self, reply: bytes) -> None:
        """Check the worker's reply and derive the session keys."""

        if self._ephemeral is None:
            raise ProtocolError("role-session handshake is already complete")
        if len(reply) not in {_REPLY_BYTES, _REPLY_BYTES + SIGNATURE_BYTES}:
            raise ProtocolError("role-session handshake reply is malformed")
        sealed = reply[:_SEALED_SECRET_BYTES]
        artifacts = reply[_SEALED_SECRET_BYTES:_REPLY_BYTES]
        signature = reply[_REPLY_BYTES:]
        middle = _hash(self._opening, self._sealed)
        try:
            worker_secret = kem.open_sealed(self._ephemeral, b"worker:" + middle, sealed)
        except kem.KemError as error:
            raise ProtocolError("role-session handshake reply does not open") from error
        transcript = _hash(middle, sealed, artifacts)
        if self._identity_public is not None:
            try:
                verify_hybrid(self._identity_public, HANDSHAKE_CONTEXT, transcript, signature)
            except SignatureError as error:
                raise ProtocolError(
                    "role-session handshake is not signed by the pinned worker identity"
                ) from error
            self.worker_signed = True
        if self._expected_artifacts is not None and not hmac.compare_digest(
            artifacts.hex(), self._expected_artifacts
        ):
            raise ProtocolError("worker loaded artifacts other than the ones deployed to it")
        keys = _derive(self._secret, worker_secret, transcript)
        context = _context(self.weight_index, self.session_id)
        self.sender = RatchetedChannel(keys.master, b"client", b"stage", context)
        self.receiver = RatchetedChannel(keys.master, b"client", b"stage", context)
        self._token = keys.token
        self.artifact_digest = artifacts.hex()
        # Dropping the ephemeral key is what makes a recorded session unopenable later.
        self._ephemeral = None
        self._secret = b""
        self.established = True

    @property
    def token_bytes(self) -> bytes:
        return self._token

    def rotate(self, epoch: int) -> None:
        if epoch <= 0:
            raise ProtocolError("role-session epoch must increase")
        self.sender.rotate(epoch)
        self.receiver.rotate(epoch)

    def data_packet(self, sealed: bytes) -> bytes:
        return DATA_MAGIC + self.session_id.encode() + sealed


def split_data_packet(packet: bytes) -> tuple[str, bytes]:
    header = len(DATA_MAGIC) + SESSION_ID_BYTES
    if len(packet) <= header or not packet.startswith(DATA_MAGIC):
        raise ProtocolError("role-session packet is malformed")
    return _session_id(packet[len(DATA_MAGIC) : header]), packet[header:]


@dataclass
class _ServerSession:
    expires_at: float
    token: bytes
    receiver: RatchetedChannel
    sender: RatchetedChannel
    epoch: int = 0


class RoleSessionStore:
    """The worker's sessions, each created by one handshake."""

    def __init__(
        self,
        recipient_key: kem.HybridKemKey,
        weight_index: int,
        *,
        artifact_digest: str,
        identity: HybridSigningKey | None = None,
        ttl_seconds: float = 300,
        capacity: int = 64,
    ) -> None:
        self._recipient = recipient_key
        self.weight_index = weight_index
        self._artifacts = bytes.fromhex(artifact_digest)
        if len(self._artifacts) != ARTIFACT_DIGEST_BYTES:
            raise ValueError("artifact digest must be a SHA-256 value")
        self._identity = identity
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._sessions: dict[str, _ServerSession] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.monotonic()
        for session_id in [key for key, value in self._sessions.items() if value.expires_at <= now]:
            self._sessions.pop(session_id, None)

    def accept(self, hello: bytes, binding: bytes = UNBOUND) -> tuple[str, bytes]:
        """Answer a client hello; return the new session's identifier and the reply."""

        if len(hello) != _HELLO_BYTES or not hello.startswith(HANDSHAKE_MAGIC):
            raise ProtocolError("role-session hello is malformed")
        offset = len(HANDSHAKE_MAGIC)
        session_id = _session_id(hello[offset : offset + SESSION_ID_BYTES])
        offset += SESSION_ID_BYTES
        ephemeral = hello[offset : offset + kem.PUBLIC_KEY_BYTES]
        sealed_client = hello[offset + kem.PUBLIC_KEY_BYTES :]
        index = self.weight_index.to_bytes(4, "big")
        opening = _hash(index, session_id.encode(), binding, self._recipient.public_key, ephemeral)
        try:
            client_secret = kem.open_sealed(self._recipient, b"client:" + opening, sealed_client)
        except kem.KemError as error:
            raise ProtocolError("role-session hello does not open") from error
        middle = _hash(opening, sealed_client)
        worker_secret = secrets.token_bytes(SECRET_BYTES)
        try:
            sealed_worker = kem.seal(ephemeral, b"worker:" + middle, worker_secret)
        except kem.KemError as error:
            raise ProtocolError("role-session hello carries an invalid key") from error
        transcript = _hash(middle, sealed_worker, self._artifacts)
        keys = _derive(client_secret, worker_secret, transcript)
        context = _context(self.weight_index, session_id)
        session = _ServerSession(
            expires_at=time.monotonic() + self.ttl_seconds,
            token=keys.token,
            receiver=RatchetedChannel(keys.master, b"stage", b"client", context),
            sender=RatchetedChannel(keys.master, b"stage", b"client", context),
        )
        with self._lock:
            self._prune()
            if session_id in self._sessions:
                # A second hello for a live session is a replay or a collision; either
                # way the existing session must not be replaced.
                raise ProtocolError("role-session identifier is already in use")
            if len(self._sessions) >= self.capacity:
                raise ProtocolError("role-session capacity is exhausted")
            self._sessions[session_id] = session
        signature = (
            b"" if self._identity is None else self._identity.sign(HANDSHAKE_CONTEXT, transcript)
        )
        return session_id, sealed_worker + self._artifacts + signature

    def channels(self, session_id: str) -> tuple[RatchetedChannel, RatchetedChannel]:
        """Return ``(receiver, sender)`` for a live session; an expired one must re-handshake."""

        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is None:
                raise ProtocolError("role session is unknown or has expired")
            existing.expires_at = time.monotonic() + self.ttl_seconds
            return existing.receiver, existing.sender

    def authorize(self, session_id: str, token: bytes) -> None:
        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is None or not hmac.compare_digest(existing.token, token):
                raise ProtocolError("role-session credentials are not authorized")

    def rotate(self, session_id: str, epoch: int) -> None:
        with self._lock:
            self._prune()
            existing = self._sessions.get(session_id)
            if existing is None or epoch != existing.epoch + 1:
                raise ProtocolError("role-session rotation epoch is invalid")
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
