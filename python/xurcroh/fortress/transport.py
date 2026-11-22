from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import (
    FIELD_MODULUS,
    MAX_ACTIVATION_BITS,
    PRIME_MODULUS,
    activation_row_seed,
    check_unreduced_product_bound,
    quantized_weight_bound,
    seeded_share_seed,
    seeded_split_secret,
)
from xurcroh.fortress.checkpoint import (
    CHECKPOINT_NONCE_BYTES,
    SessionTranscript,
    checkpoint_request,
    open_checkpoint,
)
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.grid import RoleExecutionError, WeightShare
from xurcroh.fortress.onion import seal_onion
from xurcroh.fortress.session import ClientRoleSession, channel_binding
from xurcroh.fortress.storage import DispersedShard, reconstruct_shard
from xurcroh.kem import PUBLIC_KEY_BYTES as RECIPIENT_PUBLIC_BYTES

_execution_health: ContextVar[tuple[threading.Event, float] | None] = ContextVar(
    "xurcroh_execution_health", default=None
)


@contextlib.contextmanager
def execution_health(cancelled: threading.Event, deadline: float) -> Iterator[None]:
    token = _execution_health.set((cancelled, deadline))
    try:
        yield
    finally:
        _execution_health.reset(token)


def check_execution_health() -> None:
    state = _execution_health.get()
    if state is None:
        return
    cancelled, deadline = state
    if cancelled.is_set():
        from xurcroh.errors import RequestCancelledError

        raise RequestCancelledError("request was cancelled")
    if time.monotonic() >= deadline:
        from xurcroh.errors import RequestTimeoutError

        raise RequestTimeoutError("request exceeded its execution deadline")


@dataclass(frozen=True)
class MutualTLSIdentity:
    certificates: tuple[bytes, ...]
    private_key: bytes = field(repr=False)
    pinned_ca: bytes
    server_name: str = "localhost"


@dataclass(frozen=True)
class RelayHop:
    address: str
    public_key: bytes


@dataclass(frozen=True)
class RelayRoute:
    """An onion path: entry relay, any further relays, then the role worker.

    Each relay can decrypt only its own layer, which names the next hop. The entry
    relay sees the client address and the exit relay sees the worker address.
    """

    relay_address: str
    worker_address: str
    relay_public_key: bytes
    worker_public_key: bytes
    extra_hops: tuple[RelayHop, ...] = ()

    @property
    def hop_count(self) -> int:
        return 1 + len(self.extra_hops)

    def onion_hops(self) -> list[tuple[bytes, str]]:
        relays = [RelayHop(self.relay_address, self.relay_public_key), *self.extra_hops]
        targets = [*(hop.address for hop in relays[1:]), self.worker_address]
        hops = [
            (relay.public_key, f"quic://{target}")
            for relay, target in zip(relays, targets, strict=True)
        ]
        return [*hops, (self.worker_public_key, "")]


def relay_route_from_labels(
    labels: Mapping[str, str], worker_address: str, *, worker: str
) -> RelayRoute | None:
    """Build the onion path for a role from its worker labels.

    ``onion_public`` is the worker's onion key and ``relay_path`` is a JSON list of
    ``{"address", "public_key"}`` hops, entry relay first. Both or neither must be set.
    """

    from xurcroh.errors import ConfigurationError

    onion_public = labels.get("onion_public")
    relay_path = labels.get("relay_path")
    if onion_public is None and relay_path is None:
        return None
    try:
        if onion_public is None or relay_path is None:
            raise ValueError("onion_public and relay_path must be set together")
        worker_key = bytes.fromhex(onion_public)
        document = json.loads(relay_path)
        if not isinstance(document, list) or not document:
            raise ValueError("relay_path must list at least one hop")
        hops = [RelayHop(str(hop["address"]), bytes.fromhex(hop["public_key"])) for hop in document]
        if len(worker_key) != RECIPIENT_PUBLIC_BYTES or any(
            len(hop.public_key) != RECIPIENT_PUBLIC_BYTES or not hop.address for hop in hops
        ):
            raise ValueError(
                f"onion keys must be {RECIPIENT_PUBLIC_BYTES} bytes (X25519 with ML-KEM-768) "
                "and hop addresses non-empty"
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(
            f"worker {worker} has an invalid onion relay route: {error}"
        ) from error
    return RelayRoute(
        relay_address=hops[0].address,
        worker_address=worker_address,
        relay_public_key=hops[0].public_key,
        worker_public_key=worker_key,
        extra_hops=tuple(hops[1:]),
    )


@dataclass(frozen=True)
class RoleEndpoint:
    activation_index: int
    weight_index: int
    address: str
    recipient_public: bytes
    provider_id: str
    relay: RelayRoute | None = None
    server_name: str | None = None
    generation: int = 0
    identity_public: bytes | None = None
    artifact_digest: str | None = None


@dataclass(frozen=True)
class RoleArtifact:
    weight_index: int
    recipient_public: bytes
    matrix: torch.Tensor
    provider_id: str
    replica_provider_id: str | None


DEFAULT_FRAME_BYTES = 16_384


def pad_packet(packet: bytes, frame_bytes: int) -> bytes:
    if frame_bytes < 256 or len(packet) + 8 > frame_bytes:
        raise ProtocolError("Fortress packet exceeds the configured fixed frame size")
    padding = frame_bytes - len(packet) - 8
    return len(packet).to_bytes(8, "little") + packet + secrets.token_bytes(padding)


def unpad_packet(packet: bytes, frame_bytes: int) -> bytes:
    if len(packet) != frame_bytes or len(packet) < 8:
        raise ProtocolError("Fortress fixed-size frame is invalid")
    inner_length = int.from_bytes(packet[:8], "little")
    if inner_length > frame_bytes - 8:
        raise ProtocolError("Fortress inner packet exceeds its frame")
    return packet[8 : 8 + inner_length]


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ProtocolError("role artifact byte field is invalid")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ProtocolError("role artifact byte field is invalid") from error


def encode_role_artifact(
    operation: CompiledOperation, weight_index: int, recipient_public: bytes
) -> bytes:
    if weight_index not in operation.shares:
        raise ValueError("weight share is not present in the compiled operation")
    if len(recipient_public) != RECIPIENT_PUBLIC_BYTES:
        raise ValueError(f"recipient public key must contain {RECIPIENT_PUBLIC_BYTES} bytes")
    share = operation.shares[weight_index]
    document = {
        "format": "xurcroh-fortress-role-v1",
        "weight_index": weight_index,
        "recipient_public": _encode_bytes(recipient_public),
        "provider_id": share.provider_id,
        "replica_provider_id": share.replica_provider_id,
    }
    if share.seed is not None:
        document.update(
            {
                "format": "xurcroh-fortress-role-v2",
                "shape": list(operation.quantized.dimensions),
                "seed": _encode_bytes(share.seed),
            }
        )
    else:
        if weight_index not in operation.storage:
            raise ValueError("stored weight share is not present in the compiled operation")
        stored = operation.storage[weight_index]
        bundle = stored.bundle
        document.update(
            {
                "shape": list(stored.shape),
                "storage": {
                    "threshold": bundle.threshold,
                    "nonce": _encode_bytes(bundle.nonce),
                    "aad": _encode_bytes(bundle.aad),
                    "slices": {
                        str(index): _encode_bytes(value) for index, value in bundle.slices.items()
                    },
                    "key_shares": {
                        str(index): _encode_bytes(value)
                        for index, value in bundle.key_shares.items()
                    },
                },
            }
        )
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def decode_role_artifact(
    encoded: bytes,
    *,
    attest: Callable[[int], bool] | None = None,
    allow_unattested: bool = False,
) -> RoleArtifact:
    if attest is not None and allow_unattested:
        raise ProtocolError("allow_unattested cannot override a deployment attestation policy")
    if attest is None and not allow_unattested:
        raise ProtocolError(
            "role artifact decode requires attestation evidence or an explicit "
            "allow_unattested development opt-in"
        )
    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("role artifact is not valid JSON") from error
    if not isinstance(document, dict) or document.get("format") not in {
        "xurcroh-fortress-role-v1",
        "xurcroh-fortress-role-v2",
    }:
        raise ProtocolError("unsupported Fortress role artifact")
    try:
        weight_index = int(document["weight_index"])
        shape = tuple(int(value) for value in document["shape"])
        recipient_public = _decode_bytes(document["recipient_public"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ProtocolError("role artifact fields are invalid") from error
    if (
        len(shape) != 2
        or any(value <= 0 for value in shape)
        or len(recipient_public) != RECIPIENT_PUBLIC_BYTES
    ):
        raise ProtocolError("role artifact dimensions or recipient are invalid")
    if document["format"] == "xurcroh-fortress-role-v2":
        from xurcroh.fortress.arithmetic import _seeded_uniform

        seed = _decode_bytes(document.get("seed"))
        if len(seed) != 32:
            raise ProtocolError("seed-expanded role artifact requires a 32-byte seed")
        matrix = _seeded_uniform(seed, shape)
    else:
        try:
            storage_document = document["storage"]
            bundle = DispersedShard(
                threshold=int(storage_document["threshold"]),
                nonce=_decode_bytes(storage_document["nonce"]),
                aad=_decode_bytes(storage_document["aad"]),
                slices={
                    int(index): _decode_bytes(value)
                    for index, value in storage_document["slices"].items()
                },
                key_shares={
                    int(index): _decode_bytes(value)
                    for index, value in storage_document["key_shares"].items()
                },
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ProtocolError("role artifact fields are invalid") from error
        if attest is None:
            plaintext = reconstruct_shard(bundle, bundle.slices, bundle.key_shares, lambda _: True)
        else:
            plaintext = reconstruct_shard(bundle, bundle.slices, bundle.key_shares, attest)
        expected_length = shape[0] * shape[1] * 8
        if len(plaintext) != expected_length:
            raise ProtocolError("role artifact weight length does not match its shape")
        matrix = torch.frombuffer(bytearray(plaintext), dtype=torch.int64).reshape(shape).clone()
    return RoleArtifact(
        weight_index=weight_index,
        recipient_public=recipient_public,
        matrix=matrix,
        provider_id=str(document["provider_id"]),
        replica_provider_id=document.get("replica_provider_id"),
    )


def native_role_matrix(artifact: RoleArtifact) -> Any:
    import xurcroh_quic

    return xurcroh_quic.FortressWeightShare(
        artifact.weight_index,
        artifact.matrix.shape[0],
        artifact.matrix.shape[1],
        artifact.matrix.reshape(-1).tolist(),
        [],
        [],
        PRIME_MODULUS,
        artifact.provider_id,
        artifact.replica_provider_id,
    )


def native_compiled_weight_share(weight: WeightShare) -> Any:
    import xurcroh_quic

    if weight.matrix is None:
        raise ValueError("local execution requires materialized weight shares")
    return xurcroh_quic.FortressWeightShare(
        weight.weight_index,
        weight.matrix.shape[0],
        weight.matrix.shape[1],
        weight.matrix.reshape(-1).tolist(),
        weight.challenge.r.tolist(),
        weight.challenge.v.tolist(),
        weight.challenge.prime,
        weight.provider_id,
        weight.replica_provider_id,
    )


def _new_role_session(role: RoleEndpoint, link: Any) -> ClientRoleSession:
    return ClientRoleSession(
        role.weight_index,
        role.recipient_public,
        binding=channel_binding(link, relayed=role.relay is not None),
        identity_public=role.identity_public,
        artifact_digest=role.artifact_digest,
    )


class _SharedRoleConnection:
    def __init__(
        self,
        role: RoleEndpoint,
        identity: MutualTLSIdentity,
        recover: Callable[[str, int], int] | None = None,
    ) -> None:
        self.role = role
        self.identity = identity
        self.recover_callback = recover
        self.generation = role.generation
        self.client: Any = None
        self.session: ClientRoleSession
        self.transcript: SessionTranscript
        self.completed = 0
        self.used = False
        self.call_lock = threading.Lock()
        self.connect()

    def connect(self) -> None:
        import xurcroh_quic

        if self.client is not None:
            with contextlib.suppress(ValueError):
                self.client.close()
        self.client = xurcroh_quic.TensorClient.connect_mutual(
            self.role.relay.relay_address if self.role.relay is not None else self.role.address,
            self.role.server_name or self.identity.server_name,
            self.identity.pinned_ca,
            list(self.identity.certificates),
            self.identity.private_key,
        )
        # A session is tied to the connection it was established on, so a new
        # connection always starts a new session; the handshake runs on first use.
        self.session = _new_role_session(self.role, self.client)
        self.transcript = SessionTranscript(self.session.session_id)
        self.completed = 0
        self.used = False

    def recover(self) -> None:
        try:
            self.connect()
        except ValueError:
            if self.recover_callback is None:
                raise
            self.generation = self.recover_callback(self.role.provider_id, self.generation)
            self.connect()

    def close(self) -> None:
        with self.call_lock:
            if self.client is None:
                return
            try:
                if self.used and self.session.established:
                    session_id = bytes.fromhex(self.session.session_id)
                    self.client.send(session_id, 0, 0, "cancel", "", [], self.session.token_bytes)
                    response_id, _, _, kind, _, _, _ = self.client.receive()
                    if response_id != session_id or kind != "complete":
                        raise ProtocolError("role cancellation was not acknowledged")
            except ValueError as error:
                if not any(
                    marker in str(error).lower() for marker in ("closed", "connection", "timed out")
                ):
                    raise
            finally:
                self.client.close()
                self.client = None


class _RoleTransport:
    def __init__(
        self,
        role: RoleEndpoint,
        identity: MutualTLSIdentity,
        operation_name: str,
        rotation_interval: int,
        frame_bytes: int,
        connection: _SharedRoleConnection | None = None,
    ) -> None:
        import xurcroh_quic

        self.role = role
        self._quic = xurcroh_quic
        self._operation_name = operation_name.encode("ascii")
        if not self._operation_name or len(self._operation_name) > 1024:
            raise ValueError("operation name must contain 1-1024 ASCII bytes")
        self._identity = identity
        self._connection = connection
        self.client: Any = None
        self._session: ClientRoleSession
        self._transcript: SessionTranscript
        self._completed = 0
        self._used = False
        self._connect()
        self._rotation_interval = rotation_interval
        self._frame_bytes = frame_bytes
        self._call_lock = threading.Lock()
        self._execution_lock = connection.call_lock if connection is not None else self._call_lock
        self._frame_sizes: set[int] = set()
        self.reconnect_count = 0
        self.observations: list[tuple[float, bool]] = []

    @property
    def session(self) -> ClientRoleSession:
        return self._connection.session if self._connection is not None else self._session

    @property
    def transcript(self) -> SessionTranscript:
        return self._connection.transcript if self._connection is not None else self._transcript

    def _connect(self) -> None:
        if self._connection is not None:
            self.client = self._connection.client
            return
        identity = self._identity
        if self.client is not None:
            with contextlib.suppress(ValueError):
                self.client.close()
        self.client = self._quic.TensorClient.connect_mutual(
            self.role.relay.relay_address if self.role.relay is not None else self.role.address,
            self.role.server_name or identity.server_name,
            identity.pinned_ca,
            list(identity.certificates),
            identity.private_key,
        )
        self._session = _new_role_session(self.role, self.client)
        self._transcript = SessionTranscript(self._session.session_id)
        self._completed = 0
        self._used = False

    def execute(
        self,
        activation: torch.Tensor,
        request_id: str,
        sequence: int,
        *,
        cover: bool = False,
        activation_seed: bytes | None = None,
    ) -> tuple[list[int], int, int]:
        started = time.perf_counter()
        with self._execution_lock:
            if self._connection is not None:
                self.client = self._connection.client
            try:
                result = self._execute(activation, request_id, sequence, activation_seed)
            except ValueError as error:
                if not any(
                    marker in str(error).lower() for marker in ("timed out", "connection", "closed")
                ):
                    raise
                self.reconnect_count += 1
                if self._connection is not None:
                    self._connection.recover()
                    self.client = self._connection.client
                else:
                    self._connect()
                result = self._execute(activation, request_id, sequence, activation_seed)
        self.observations.append((time.perf_counter() - started, cover))
        return result

    def _execute(
        self,
        activation: torch.Tensor,
        request_id: str,
        sequence: int,
        activation_seed: bytes | None = None,
    ) -> tuple[list[int], int, int]:
        canonical = (
            activation.detach()
            .cpu()
            .to(torch.int64)
            .remainder(FIELD_MODULUS)
            .to(torch.int32)
            .contiguous()
        )
        if activation_seed is not None:
            if len(activation_seed) != 32:
                raise ValueError("seeded activation transport requires a 32-byte seed")
            operation_payload = (
                b"XF2\x01"
                + len(self._operation_name).to_bytes(2, "little")
                + self._operation_name
                + activation_seed
            )
        else:
            operation_payload = (
                len(self._operation_name).to_bytes(2, "little")
                + self._operation_name
                + canonical.numpy().tobytes()
            )
        payload, sent, received = self._round_trip(request_id, sequence, operation_payload)
        if not payload or len(payload) % 4:
            raise ProtocolError("role result must contain canonical 32-bit field elements")
        result = [
            int.from_bytes(payload[offset : offset + 4], "little", signed=False)
            for offset in range(0, len(payload), 4)
        ]
        return result, sent, received

    def checkpoint(self) -> dict[str, Any]:
        """Ask the role to sign what it has seen this session and check it against our view."""

        role = self.role
        if role.identity_public is None:
            raise ProtocolError(f"role {role.provider_id} has no pinned identity for a checkpoint")
        nonce = secrets.token_bytes(CHECKPOINT_NONCE_BYTES)
        with self._execution_lock:
            if self._connection is not None:
                self.client = self._connection.client
            # Copy our view first: the exchange below is itself added to the transcript.
            seen = self.transcript.snapshot()
            try:
                payload, _, _ = self._round_trip(
                    f"checkpoint:{uuid.uuid4().hex}", 0, checkpoint_request(nonce)
                )
            except ValueError as error:
                raise ProtocolError(
                    f"role {role.provider_id} did not return a checkpoint: {error}"
                ) from error
        return open_checkpoint(
            payload,
            worker=role.provider_id,
            identity_public=role.identity_public,
            transcript=seen,
            weight_index=role.weight_index,
            nonce=nonce,
            expected_artifacts=role.artifact_digest,
        )

    def _round_trip(
        self, request_id: str, sequence: int, operation_payload: bytes
    ) -> tuple[bytes, int, int]:
        session = self.session
        if not session.established:
            # One round trip, in the same fixed-size frames as every other call.
            handshake_id = hashlib.sha256(f"handshake:{session.session_id}".encode()).digest()[:16]
            reply, _, _ = self._exchange(handshake_id, 0, session.hello())
            session.finish(reply)
        # The identifier and sequence number travel outside the session encryption, where
        # a relay and the worker both read them. A fresh random identifier and a constant
        # sequence number tell them nothing about which operation, row or request this is,
        # nor whether it is a cover call.
        del request_id, sequence
        wire_id = secrets.token_hex(16)
        request_hash = hashlib.sha256(wire_id.encode()).digest()[:16]
        sealed = session.sender.seal(wire_id, "u8", [len(operation_payload)], operation_payload)
        sealed_response, sent, received = self._exchange(
            request_hash, 0, session.data_packet(sealed)
        )
        result_dtype, result_shape, payload = session.receiver.open_tensor(sealed_response)
        if result_dtype != "u8" or result_shape != [len(payload)]:
            raise ProtocolError("role result must be a byte vector")
        self.transcript.update(sealed, sealed_response)
        if self._connection is not None:
            self._connection.completed += 1
            completed = self._connection.completed
        else:
            self._completed += 1
            completed = self._completed
        if completed % self._rotation_interval == 0:
            session.rotate(completed // self._rotation_interval)
        return bytes(payload), sent, received

    def _exchange(self, request_hash: bytes, sequence: int, inner: bytes) -> tuple[bytes, int, int]:
        """Send one fixed-size frame and return the unpadded reply with byte counts."""

        packet = pad_packet(inner, self._frame_bytes)
        if self.role.relay is not None:
            packet = seal_onion(self.role.relay.onion_hops(), packet)
        sent = len(
            self._quic.encode_frame(
                request_hash, 0, sequence, "tensor", "u8", [len(packet)], packet
            )
        )
        self.client.send(request_hash, 0, sequence, "tensor", "u8", [len(packet)], packet)
        self._frame_sizes.add(len(packet))
        self._used = True
        if self._connection is not None:
            self._connection.used = True
        response_id, _, _, kind, dtype, shape, response = self.client.receive()
        received = len(
            self._quic.encode_frame(response_id, 0, sequence, kind, dtype or "", shape, response)
        )
        if response_id != request_hash or kind != "tensor" or dtype != "u8":
            raise ProtocolError(f"role {self.role.provider_id} returned an invalid response")
        if shape != [self._frame_bytes] or len(response) != self._frame_bytes:
            raise ProtocolError("role response tensor descriptor is invalid")
        self._frame_sizes.add(len(response))
        return unpad_packet(response, self._frame_bytes), sent, received

    def close(self) -> None:
        if self._connection is not None:
            return
        with self._execution_lock:
            try:
                if self._used and self.session.established:
                    session_id = bytes.fromhex(self.session.session_id)
                    self.client.send(session_id, 0, 0, "cancel", "", [], self.session.token_bytes)
                    response_id, _, _, kind, _, _, _ = self.client.receive()
                    if response_id != session_id or kind != "complete":
                        raise ProtocolError("role cancellation was not acknowledged")
            finally:
                if self._connection is None:
                    self.client.close()


def combine_role_results(
    activation_shares: list[torch.Tensor],
    weight_shares: dict[int, WeightShare],
    results: dict[tuple[int, int], list[int]],
    providers: dict[tuple[int, int], str],
) -> torch.Tensor:
    import xurcroh_quic

    expected = {
        (activation_index, weight_index)
        for activation_index in range(len(activation_shares))
        for weight_index in weight_shares
    }
    if set(results) != expected or not results:
        raise ProtocolError("grid response roles are incomplete or duplicated")
    output_size = len(next(iter(results.values())))
    if any(len(product) != output_size for product in results.values()):
        raise ProtocolError("grid response dimensions are inconsistent")
    for activation_index, activation in enumerate(activation_shares):
        canonical = activation.remainder(FIELD_MODULUS)
        for weight_index, share in weight_shares.items():
            key = (activation_index, weight_index)
            exact_product = results[key]
            prime = share.challenge.prime
            if any(value < 0 or value >= prime for value in exact_product):
                raise RoleExecutionError(
                    activation_index,
                    weight_index,
                    providers[key],
                    "worker returned a non-canonical field result",
                )
            product = torch.tensor(exact_product, dtype=torch.int64)
            if not all(
                xurcroh_quic.verify_fortress_product(
                    canonical.tolist(),
                    product.tolist(),
                    challenge.r.tolist(),
                    challenge.v.tolist(),
                    prime,
                )
                for challenge in share.challenges
            ):
                raise RoleExecutionError(
                    activation_index,
                    weight_index,
                    providers[key],
                    "Freivalds check failed",
                )
    ordered = [results[key] for key in sorted(expected)]
    return torch.tensor(
        xurcroh_quic.combine_field_products(ordered, FIELD_MODULUS), dtype=torch.int64
    )


def execute_native_grid(
    activation_shares: list[torch.Tensor], weight_shares: dict[int, WeightShare]
) -> torch.Tensor:
    import xurcroh_quic

    if activation_shares and activation_shares[0].ndim == 2:
        outputs = [
            execute_native_grid([share[row] for share in activation_shares], weight_shares)
            for row in range(activation_shares[0].shape[0])
        ]
        return torch.stack(outputs)
    results: dict[tuple[int, int], list[int]] = {}
    providers: dict[tuple[int, int], str] = {}
    for activation_index, activation in enumerate(activation_shares):
        canonical = activation.remainder(FIELD_MODULUS).tolist()
        for weight_index, share in weight_shares.items():
            key = (activation_index, weight_index)
            results[key] = xurcroh_quic.execute_fortress_role(
                canonical, native_compiled_weight_share(share)
            )
            providers[key] = share.provider_id
    return combine_role_results(activation_shares, weight_shares, results, providers)


class _SlotClock:
    """Hands out evenly spaced send slots, so calls leave at a constant rate.

    A real call takes the next free slot. A cover call takes a slot only when no real
    call is waiting for one, so real calls replace cover calls instead of adding to them.
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = time.monotonic() + interval
        self._waiting = 0

    def wait_for_slot(self) -> None:
        with self._lock:
            self._waiting += 1
            slot = max(time.monotonic(), self._next)
            self._next = slot + self.interval
        try:
            delay = slot - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        finally:
            with self._lock:
                self._waiting -= 1

    def take_idle_slot(self) -> float:
        """Take the current slot for a cover call, or return how long to wait and retry."""

        with self._lock:
            now = time.monotonic()
            if self._waiting:
                return self.interval / 4
            if now < self._next:
                return self._next - now
            self._next = max(self._next + self.interval, now + self.interval / 2)
            return 0.0


def _cover_failure_is_fatal(error: Exception) -> bool:
    """A wrong answer or a broken protocol stops everything; a lost link does not."""

    return isinstance(error, (RoleExecutionError, ProtocolError))


class FortressGridRuntime:
    def __init__(
        self,
        roles: list[RoleEndpoint],
        identity: MutualTLSIdentity,
        *,
        rotation_interval: int = 64,
        frame_bytes: int = DEFAULT_FRAME_BYTES,
        cover_interval: float = 1.0,
        seeded_activations: bool = True,
        recover_role: Callable[[str, int], int] | None = None,
        constant_rate: bool = False,
    ) -> None:
        if not roles or cover_interval <= 0:
            raise ValueError("a Fortress grid requires roles and a positive cover interval")
        coordinates = {(role.activation_index, role.weight_index) for role in roles}
        if len(coordinates) != len(roles):
            raise ValueError("a Fortress grid cannot contain duplicate role coordinates")
        self.executor = ThreadPoolExecutor(max_workers=len(roles))
        self.connections = {
            (role.activation_index, role.weight_index): _SharedRoleConnection(
                role, identity, recover_role
            )
            for role in roles
        }
        self.rotation_interval = rotation_interval
        self.frame_bytes = frame_bytes
        self.cover_interval = cover_interval
        self.seeded_activations = seeded_activations
        self.constant_rate = constant_rate
        self._clock = _SlotClock(cover_interval) if constant_rate else None
        self._clients: list[FortressRoleClient] = []
        self._lock = threading.Lock()
        self._closed = False
        self._cover_error: Exception | None = None
        self.cover_failure_count = 0
        self._cover_stop = threading.Event()
        self._cover_thread = threading.Thread(target=self._cover_loop, daemon=True)
        self._cover_thread.start()

    def client(self, operation: CompiledOperation, operation_name: str) -> FortressRoleClient:
        return FortressRoleClient(
            operation,
            list(self._roles()),
            self._identity(),
            operation_name=operation_name,
            runtime=self,
        )

    def _roles(self) -> list[RoleEndpoint]:
        return [connection.role for _, connection in sorted(self.connections.items())]

    def _identity(self) -> MutualTLSIdentity:
        connection = next(iter(self.connections.values()))
        return connection.identity

    def _register(self, client: FortressRoleClient) -> None:
        with self._lock:
            if self._closed:
                raise ProtocolError("Fortress grid runtime is closed")
            self._clients.append(client)

    def _unregister(self, client: FortressRoleClient) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def _cover_loop(self) -> None:
        index = 0
        while not self._cover_stop.is_set():
            if self._clock is None:
                if self._cover_stop.wait(self.cover_interval):
                    return
            else:
                delay = self._clock.take_idle_slot()
                if delay > 0:
                    self._cover_stop.wait(delay)
                    continue
            with self._lock:
                clients = list(self._clients)
            if not clients:
                continue
            client = clients[index % len(clients)]
            index += 1
            try:
                client._send_cover()
            except Exception as error:
                client.cover_failure_count += 1
                self.cover_failure_count += 1
                if _cover_failure_is_fatal(error):
                    client._cover_error = error
                    self._cover_error = error
                    return

    def checkpoints(self) -> list[dict[str, Any]]:
        """A verified, worker-signed checkpoint for every role of this grid, in role order."""

        with self._lock:
            if not self._clients:
                raise ProtocolError("a checkpoint requires an open client on this grid")
            client = self._clients[0]
        return client.checkpoints()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clients.clear()
        self._cover_stop.set()
        self._cover_thread.join(timeout=30)
        self.executor.shutdown(wait=True)
        first_error = None
        for connection in self.connections.values():
            try:
                connection.close()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise ProtocolError("Fortress grid runtime did not close cleanly") from first_error


class FortressRoleClient:
    def __init__(
        self,
        operation: CompiledOperation,
        roles: list[RoleEndpoint],
        identity: MutualTLSIdentity,
        *,
        operation_name: str = "operation",
        rotation_interval: int = 64,
        frame_bytes: int = DEFAULT_FRAME_BYTES,
        cover_interval: float = 1.0,
        runtime: FortressGridRuntime | None = None,
    ) -> None:
        if rotation_interval <= 0:
            raise ValueError("rotation interval must be positive")
        if frame_bytes < 256 or cover_interval <= 0:
            raise ValueError("frame size and cover interval must be positive")
        self.operation = operation
        self.roles = {(role.activation_index, role.weight_index): role for role in roles}
        expected = {
            (activation_index, weight_index)
            for activation_index in range(len(operation.shares))
            for weight_index in operation.shares
        }
        if len(roles) != len(self.roles) or set(self.roles) != expected:
            raise ValueError("each Fortress grid role requires exactly one endpoint")
        self._providers = {key: role.provider_id for key, role in self.roles.items()}
        self._runtime = runtime
        self.seeded_activations = runtime.seeded_activations if runtime is not None else False
        self._executor = (
            runtime.executor if runtime is not None else ThreadPoolExecutor(max_workers=len(roles))
        )
        self._transports = {
            key: _RoleTransport(
                role,
                identity,
                operation_name,
                runtime.rotation_interval if runtime is not None else rotation_interval,
                runtime.frame_bytes if runtime is not None else frame_bytes,
                runtime.connections[key] if runtime is not None else None,
            )
            for key, role in self.roles.items()
        }
        self._lock = threading.Lock()
        self._closed = False
        self.request_count = 0
        self.verified_result_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.cover_request_count = 0
        self.cover_bytes_sent = 0
        self.cover_bytes_received = 0
        self._cover_interval = cover_interval
        self._cover_stop = threading.Event()
        self._cover_error: Exception | None = None
        self.cover_failure_count = 0
        self._quarantined: RoleExecutionError | None = None
        self._cover_thread = threading.Thread(target=self._cover_loop, daemon=True)
        if runtime is None:
            self._cover_thread.start()
        else:
            runtime._register(self)

    @property
    def frame_sizes(self) -> set[int]:
        sizes: set[int] = set()
        for transport in self._transports.values():
            sizes.update(transport._frame_sizes)
        return sizes

    @property
    def timing_observations(self) -> list[tuple[float, bool]]:
        return [
            observation
            for transport in self._transports.values()
            for observation in transport.observations
        ]

    def _execute_token(
        self,
        activation_shares: list[torch.Tensor],
        request_id: str,
        sequence: int,
        *,
        cover: bool = False,
        activation_seed: bytes | None = None,
    ) -> torch.Tensor:
        clock = self._runtime._clock if self._runtime is not None else None
        if clock is not None and not cover:
            clock.wait_for_slot()
        futures = {
            key: self._executor.submit(
                self._transports[key].execute,
                activation_shares[key[0]],
                request_id,
                sequence,
                cover=cover,
                activation_seed=(
                    seeded_share_seed(activation_seed, key[0])
                    if activation_seed is not None and key[0] < len(activation_shares) - 1
                    else None
                ),
            )
            for key in self._transports
        }
        results: dict[tuple[int, int], list[int]] = {}
        for key, future in futures.items():
            result, sent, received = future.result()
            results[key] = result
            with self._lock:
                if cover:
                    self.cover_request_count += 1
                    self.cover_bytes_sent += sent
                    self.cover_bytes_received += received
                else:
                    self.request_count += 1
                    self.bytes_sent += sent
                    self.bytes_received += received
        try:
            combined = combine_role_results(
                activation_shares, self.operation.shares, results, self._providers
            )
        except RoleExecutionError as error:
            with self._lock:
                self._quarantined = self._quarantined or error
            raise
        if not cover:
            with self._lock:
                self.verified_result_count += len(results)
        return combined

    def checkpoints(self) -> list[dict[str, Any]]:
        """A verified, worker-signed checkpoint for every role session of this client."""

        return [self._transports[key].checkpoint() for key in sorted(self._transports)]

    def _send_cover(self) -> None:
        """One call on random data, built exactly as a real call is built.

        With seeded activations a real call gives every share but the last as a 32-byte
        seed; the cover call does the same, so a role sees the same payload either way.
        The last share of a real call is uniformly random to its holder, as here.
        """

        features = self.operation.quantized.dimensions[1]
        count = len(self.operation.shares)
        if self.seeded_activations:
            seed = secrets.token_bytes(32)
            shares = seeded_split_secret(
                torch.randint(0, FIELD_MODULUS, (features,), dtype=torch.int64),
                seed=seed,
                count=count,
            ).shares
            self._execute_token(shares, "cover", 0, cover=True, activation_seed=seed)
            return
        shares = [
            torch.randint(0, FIELD_MODULUS, (features,), dtype=torch.int64) for _ in range(count)
        ]
        self._execute_token(shares, "cover", 0, cover=True)

    def _cover_loop(self) -> None:
        while not self._cover_stop.wait(self._cover_interval):
            try:
                self._send_cover()
            except Exception as error:
                self.cover_failure_count += 1
                if _cover_failure_is_fatal(error):
                    self._cover_error = error
                    return

    def execute(
        self,
        activation_shares: list[torch.Tensor],
        *,
        request_id: str | None = None,
        activation_seed: bytes | None = None,
    ) -> torch.Tensor:
        if activation_seed is not None and (
            not self.seeded_activations or len(activation_seed) != 32
        ):
            raise ValueError("a 32-byte activation seed requires seeded grid execution")
        with self._lock:
            if self._closed:
                raise ProtocolError("Fortress role client is closed")
            if self._quarantined is not None:
                raise ProtocolError("Fortress role client is quarantined") from self._quarantined
            if self._cover_error is not None:
                raise ProtocolError("Fortress cover traffic failed") from self._cover_error
            if self._runtime is not None and self._runtime._cover_error is not None:
                error = self._runtime._cover_error
                raise ProtocolError(
                    f"Fortress grid cover traffic failed: {type(error).__name__}: {error}"
                ) from error
        if len(activation_shares) != len(self.operation.shares):
            raise ValueError("activation share count must match the compiled weight grid")
        correlation_id = request_id or uuid.uuid4().hex
        vector_input = activation_shares[0].ndim == 1
        rows_by_share = [
            share.reshape(1, -1) if share.ndim == 1 else share for share in activation_shares
        ]
        rows = rows_by_share[0]
        if any(share.shape != rows.shape for share in rows_by_share):
            raise ValueError("activation share shapes must match")
        check_unreduced_product_bound(
            rows.shape[1],
            2 ** (MAX_ACTIVATION_BITS - 1),
            quantized_weight_bound(self.operation.quantized),
        )
        outputs = []
        for row in range(rows.shape[0]):
            check_execution_health()
            outputs.append(
                self._execute_token(
                    [share[row] for share in rows_by_share],
                    correlation_id,
                    row,
                    activation_seed=(
                        activation_row_seed(activation_seed, row)
                        if activation_seed is not None
                        else None
                    ),
                )
            )
        check_execution_health()
        return outputs[0] if vector_input else torch.stack(outputs)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._cover_stop.set()
        if self._runtime is None:
            self._cover_thread.join(timeout=30)
            self._executor.shutdown(wait=True)
        else:
            self._runtime._unregister(self)
        first_error: Exception | None = None
        for transport in self._transports.values():
            try:
                transport.close()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise ProtocolError("one or more Fortress roles did not close cleanly") from first_error
