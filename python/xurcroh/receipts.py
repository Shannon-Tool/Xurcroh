"""Signed per-task traces: what was requested, what ran, and how it was protected.

A trace is produced by the trusted client for every completed request. It holds no
prompt or output text, only salted commitments to them, so it can be stored and
shared without revealing content. The signature shows the trace was not changed
after the client produced it; it does not by itself make a remote operator honest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.errors import ConfigurationError, XurcrohError
from xurcroh.merkle_log import (
    MerkleLog,
    MerkleProofError,
    leaf_hash,
    verify_inclusion,
    verify_tree_head,
)

TRACE_FORMAT = "xurcroh-task-trace-v1"
SAMPLER = "xurcroh-multinomial-v2"
SIGNATURE_ALGORITHM = "Ed25519"


class TraceError(XurcrohError):
    """A trace failed verification; the message names the part that failed."""


def canonical(document: Any) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def canonical_messages(messages: Sequence[Any]) -> bytes:
    from xurcroh.requests import InputMessage

    normalized = [
        (
            message if isinstance(message, InputMessage) else InputMessage.model_validate(message)
        ).model_dump(mode="json")
        for message in messages
    ]
    return canonical(normalized)


def canonical_media(media_input: Mapping[str, Any] | None) -> bytes:
    """Canonical form of an image or video request's prompt, input image and mask."""

    values = dict(media_input or {})
    return canonical({name: values.get(name) for name in ("prompt", "image", "mask")})


def commit(payload: bytes, salt: bytes | None = None) -> dict[str, str]:
    """Return a salted SHA-256 commitment that hides ``payload`` without the salt."""

    salt = secrets.token_bytes(16) if salt is None else salt
    return {"salt": salt.hex(), "sha256": hashlib.sha256(salt + payload).hexdigest()}


def _opens(commitment: Any, payload: bytes) -> bool:
    try:
        expected = commit(payload, bytes.fromhex(commitment["salt"]))["sha256"]
        return hmac.compare_digest(expected, str(commitment["sha256"]))
    except (KeyError, TypeError, ValueError):
        return False


def public_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def sign_trace(document: Mapping[str, Any], signing_key: Ed25519PrivateKey) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "signature"}
    body["format"] = TRACE_FORMAT
    public_key = public_key_bytes(signing_key)
    body["signer"] = {"algorithm": SIGNATURE_ALGORITHM, "public_key": public_key.hex()}
    return {**body, "signature": signing_key.sign(canonical(body)).hex()}


def verify_trace(
    document: Mapping[str, Any],
    *,
    public_key: bytes | None = None,
    messages: Sequence[Any] | None = None,
    media_input: Mapping[str, Any] | None = None,
    output: str | None = None,
) -> dict[str, bool | None]:
    """Check a trace's signature and, when given, that it refers to this prompt and output.

    ``public_key`` pins the signer. Without it the key embedded in the trace is used,
    which detects accidental damage but not a trace re-signed by someone else.
    """

    if document.get("format") != TRACE_FORMAT:
        raise TraceError(f"trace format is not {TRACE_FORMAT}")
    body = {key: value for key, value in document.items() if key != "signature"}
    try:
        signer = document["signer"]
        embedded = bytes.fromhex(signer["public_key"])
        signature = bytes.fromhex(document["signature"])
        if signer["algorithm"] != SIGNATURE_ALGORITHM:
            raise ValueError("unsupported algorithm")
        key = embedded if public_key is None else public_key
        if not hmac.compare_digest(key, embedded):
            raise ValueError("signer does not match the expected key")
        Ed25519PublicKey.from_public_bytes(key).verify(signature, canonical(body))
    except (KeyError, TypeError, ValueError, InvalidSignature) as error:
        raise TraceError("trace signature is missing or does not verify") from error
    report: dict[str, bool | None] = {"signature": True, "prompt": None, "output": None}
    if messages is not None and media_input is not None:
        raise TraceError("give chat messages or a media input to check the prompt, not both")
    if messages is not None or media_input is not None:
        payload = (
            canonical_messages(messages) if messages is not None else canonical_media(media_input)
        )
        if not _opens(document.get("request", {}).get("prompt"), payload):
            raise TraceError("trace does not refer to this prompt")
        report["prompt"] = True
    if output is not None:
        if not _opens(document.get("result", {}).get("output"), output.encode()):
            raise TraceError("trace does not refer to this output")
        report["output"] = True
    return report


def trace_digest(document: Mapping[str, Any]) -> bytes:
    """The value logged for a trace: a hash that reveals none of its contents."""

    return hashlib.sha256(canonical(document)).digest()


def verify_trace_proof(
    document: Mapping[str, Any], proof: Mapping[str, Any], *, log_key: bytes
) -> int:
    """Check that exactly this trace is in the log signed by ``log_key``; return the tree size.

    A trace changed after it was logged no longer matches its logged hash.
    """

    try:
        size, root = verify_tree_head(proof["tree_head"], log_key)
        verify_inclusion(
            leaf_hash(trace_digest(document)),
            int(proof["leaf_index"]),
            size,
            [bytes.fromhex(node) for node in proof["path"]],
            root,
        )
    except (KeyError, TypeError, ValueError, MerkleProofError) as error:
        raise TraceError(f"trace is not in the log: {error}") from error
    return size


def describe_trace(document: Mapping[str, Any]) -> list[str]:
    """Render the facts of a verified trace as plain lines for a person to read."""

    lines = [f"task: {document.get('task_id')}", f"mode: {document.get('mode')}"]
    for section in ("model", "security", "transport"):
        for name, value in sorted(dict(document.get(section, {})).items()):
            lines.append(f"{section}.{name}: {json.dumps(value, sort_keys=True)}")
    request = dict(document.get("request", {}))
    for group in ("settings", "effective"):
        for name, value in sorted(dict(request.get(group, {})).items()):
            lines.append(f"request.{group}.{name}: {json.dumps(value)}")
    for name, value in sorted(dict(document.get("result", {})).items()):
        if name != "output":
            lines.append(f"result.{name}: {json.dumps(value)}")
    return lines


class TraceStore:
    """Keep the most recent signed traces in memory, keyed by task ID."""

    def __init__(self, limit: int, signing_key: Ed25519PrivateKey | None = None) -> None:
        self._limit = limit
        self._key = signing_key or Ed25519PrivateKey.generate()
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._indexes: dict[str, int] = {}
        self._log = MerkleLog(Ed25519PrivateKey.generate())
        self._lock = threading.Lock()

    @property
    def log_key(self) -> bytes:
        return self._log.public_key

    def proof(self, task_id: str) -> dict[str, Any]:
        """Inclusion proof for a held trace against the current signed tree head."""

        self.get(task_id)
        with self._lock:
            index = self._indexes[task_id]
            head = self._log.signed_tree_head()
            path = self._log.inclusion_proof(index, head["tree_size"])
        return {
            "leaf_index": index,
            "path": [node.hex() for node in path],
            "tree_head": head,
        }

    @property
    def public_key(self) -> bytes:
        return public_key_bytes(self._key)

    def record(self, document: Mapping[str, Any]) -> dict[str, Any]:
        signed = sign_trace(document, self._key)
        task_id = str(signed["task_id"])
        with self._lock:
            self._indexes[task_id] = self._log.append(trace_digest(signed))
            self._traces[task_id] = signed
            self._traces.move_to_end(task_id)
            while len(self._traces) > self._limit:
                dropped, _ = self._traces.popitem(last=False)
                self._indexes.pop(dropped, None)
        return signed

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            trace = self._traces.get(task_id)
        if trace is None:
            raise ConfigurationError(
                f"no trace is held for task {task_id!r}; traces are kept for the most recent "
                f"{self._limit} tasks of this client"
            )
        return json.loads(canonical(trace))  # type: ignore[no-any-return]
