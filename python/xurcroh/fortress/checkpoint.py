"""Worker-signed session checkpoints.

A role worker holds a long-lived hybrid identity key. On request it signs a
statement of which artifacts it loaded and a running hash of every sealed
request and response of the session so far. The client keeps the same running
hash, so a checkpoint that verifies shows the named worker saw and answered
exactly the calls the client made, with the artifacts the deployment pinned.

A checkpoint does not show the worker computed correctly (the result check does
that) and does not show where the worker runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any

from xurcroh.errors import ProtocolError
from xurcroh.signatures import (
    SIGNATURE_BYTES,
    HybridSigningKey,
    SignatureError,
    key_fingerprint,
    verify_hybrid,
)

CHECKPOINT_OPERATION = "xurcroh.checkpoint"
CHECKPOINT_FORMAT = "xurcroh-worker-checkpoint-v1"
CHECKPOINT_CONTEXT = b"xurcroh-worker-checkpoint-v1"
CHECKPOINT_NONCE_BYTES = 32
_TRANSCRIPT_DOMAIN = b"xurcroh-session-transcript-v1"
_ARTIFACT_DOMAIN = b"xurcroh-role-artifacts-v1"


def _canonical(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def artifact_set_digest(artifacts: Mapping[str, bytes]) -> str:
    """One digest for every operation artifact a role holds, independent of load order."""

    return artifact_set_digest_from_hashes(
        {name: hashlib.sha256(artifact).hexdigest() for name, artifact in artifacts.items()}
    )


def artifact_set_digest_from_hashes(hashes: Mapping[str, str]) -> str:
    """The same digest, from each artifact's SHA-256 so the bytes need not be held."""

    entries = [[name, hashes[name]] for name in sorted(hashes)]
    return hashlib.sha256(_ARTIFACT_DOMAIN + _canonical(entries)).hexdigest()


class SessionTranscript:
    """Running hash over the sealed request and response bytes of one session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.calls = 0
        self._state = hashlib.sha256(_TRANSCRIPT_DOMAIN + session_id.encode()).digest()

    def update(self, request: bytes, response: bytes) -> None:
        self._state = hashlib.sha256(
            self._state
            + len(request).to_bytes(8, "big")
            + request
            + len(response).to_bytes(8, "big")
            + response
        ).digest()
        self.calls += 1

    def hexdigest(self) -> str:
        return self._state.hex()

    def snapshot(self) -> SessionTranscript:
        copy = SessionTranscript(self.session_id)
        copy.calls, copy._state = self.calls, self._state
        return copy


def checkpoint_request(nonce: bytes) -> bytes:
    name = CHECKPOINT_OPERATION.encode()
    return len(name).to_bytes(2, "little") + name + nonce


def checkpoint_nonce(payload: bytes) -> bytes | None:
    """Return the nonce when ``payload`` is a checkpoint request, otherwise ``None``."""

    prefix = checkpoint_request(b"")
    if not payload.startswith(prefix):
        return None
    nonce = payload[len(prefix) :]
    if len(nonce) != CHECKPOINT_NONCE_BYTES:
        raise ProtocolError("checkpoint request requires a 32-byte nonce")
    return nonce


def sign_checkpoint(
    key: HybridSigningKey,
    transcript: SessionTranscript,
    *,
    artifacts: str,
    weight_index: int,
    nonce: bytes,
) -> bytes:
    body = _canonical(
        {
            "format": CHECKPOINT_FORMAT,
            "session": transcript.session_id,
            "calls": transcript.calls,
            "transcript": transcript.hexdigest(),
            "artifacts": artifacts,
            "weight_index": weight_index,
            "nonce": nonce.hex(),
            "identity": key_fingerprint(key.public_key),
        }
    )
    return len(body).to_bytes(4, "little") + body + key.sign(CHECKPOINT_CONTEXT, body)


def verify_checkpoint_signature(document: Mapping[str, Any]) -> dict[str, Any]:
    """Check a stored checkpoint against the identity key recorded with it; return its body.

    This shows the holder of that key signed the body. Whether the key belongs to
    the expected worker is decided by whoever pinned it.
    """

    try:
        body = dict(document["body"])
        public = bytes.fromhex(document["identity_public"])
        signature = bytes.fromhex(document["signature"])
        if body["format"] != CHECKPOINT_FORMAT:
            raise ValueError("unknown checkpoint format")
        if body["identity"] != key_fingerprint(public):
            raise ValueError("checkpoint names a different identity key")
        verify_hybrid(public, CHECKPOINT_CONTEXT, _canonical(body), signature)
    except (KeyError, TypeError, ValueError, SignatureError) as error:
        raise ProtocolError(f"worker checkpoint signature does not verify: {error}") from error
    return body


def open_checkpoint(
    payload: bytes,
    *,
    worker: str,
    identity_public: bytes,
    transcript: SessionTranscript,
    weight_index: int,
    nonce: bytes,
    expected_artifacts: str | None,
) -> dict[str, Any]:
    """Verify a checkpoint reply against the client's own view of the session."""

    def reject(reason: str) -> ProtocolError:
        return ProtocolError(f"role {worker} checkpoint rejected: {reason}")

    if len(payload) < 4:
        raise reject("reply is truncated")
    length = int.from_bytes(payload[:4], "little")
    raw, signature = payload[4 : 4 + length], payload[4 + length :]
    if len(raw) != length or len(signature) != SIGNATURE_BYTES:
        raise reject("reply is malformed")
    try:
        verify_hybrid(identity_public, CHECKPOINT_CONTEXT, raw, signature)
    except SignatureError as error:
        raise reject("signature does not match the pinned worker identity") from error
    try:
        body = json.loads(raw)
    except ValueError as error:
        raise reject("body is not JSON") from error
    if not isinstance(body, dict) or _canonical(body) != raw:
        raise reject("body is not canonical")
    expected: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "identity": key_fingerprint(identity_public),
        "nonce": nonce.hex(),
        "session": transcript.session_id,
        "weight_index": weight_index,
        "calls": transcript.calls,
        "transcript": transcript.hexdigest(),
    }
    reasons = {
        "nonce": "reply is not fresh",
        "session": "reply is for another session",
        "calls": "worker counted a different number of calls",
        "transcript": "worker saw different traffic than the client sent",
    }
    for name, value in expected.items():
        if body.get(name) != value:
            raise reject(reasons.get(name, f"unexpected {name}"))
    artifacts = body.get("artifacts")
    if not isinstance(artifacts, str):
        raise reject("artifact digest is missing")
    if expected_artifacts is not None and not hmac.compare_digest(artifacts, expected_artifacts):
        raise reject("worker loaded artifacts other than the ones deployed to it")
    if set(body) != {*expected, "artifacts"}:
        raise reject("body has unexpected fields")
    return {
        "worker": worker,
        "body": body,
        "signature": signature.hex(),
        "identity_public": identity_public.hex(),
        "artifacts_pinned": expected_artifacts is not None,
    }
