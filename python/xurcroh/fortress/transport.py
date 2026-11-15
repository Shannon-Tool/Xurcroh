from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import (
    FIELD_MODULUS,
    PRIME_MODULUS,
    activation_row_seed,
    seeded_share_seed,
)
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.grid import RoleExecutionError, WeightShare
from xurcroh.fortress.storage import DispersedShard, reconstruct_shard
from xurcroh.runtime.stage_sessions import ClientStageSession


@dataclass(frozen=True)
class MutualTLSIdentity:
    certificates: tuple[bytes, ...]
    private_key: bytes = field(repr=False)
    pinned_ca: bytes
    server_name: str = "localhost"


@dataclass(frozen=True)
class RelayRoute:
    relay_address: str
    worker_address: str
    relay_public_key: bytes
    worker_public_key: bytes


@dataclass(frozen=True)
class RoleEndpoint:
    activation_index: int
    weight_index: int
    address: str
    recipient_public: bytes
    provider_id: str
    relay: RelayRoute | None = None
    server_name: str | None = None


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
    if len(recipient_public) != 32:
        raise ValueError("recipient public key must contain 32 bytes")
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


def decode_role_artifact(encoded: bytes) -> RoleArtifact:
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
    if len(shape) != 2 or any(value <= 0 for value in shape) or len(recipient_public) != 32:
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
        plaintext = reconstruct_shard(bundle, bundle.slices, bundle.key_shares, lambda _: True)
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


class _SharedRoleConnection:
    def __init__(
        self,
        role: RoleEndpoint,
        identity: MutualTLSIdentity,
        recover: Callable[[str], None] | None = None,
    ) -> None:
        self.role = role
        self.identity = identity
        self.recover_callback = recover
        self.client: Any = None
        self.session = ClientStageSession(role.weight_index, role.recipient_public)
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

    def recover(self) -> None:
        try:
            self.connect()
        except ValueError:
            if self.recover_callback is None:
                raise
            self.recover_callback(self.role.provider_id)
            self.connect()
        self.session = ClientStageSession(self.role.weight_index, self.role.recipient_public)
        self.completed = 0
        self.used = False

    def close(self) -> None:
        with self.call_lock:
            if self.client is None:
                return
            try:
                if self.used:
                    session_id = bytes.fromhex(self.session.session_id)
                    self.client.send(session_id, 0, 0, "cancel", "", [], self.session.token_bytes)
                    response_id, _, _, kind, _, _, _ = self.client.receive()
                    if response_id != session_id or kind != "complete":
                        raise ProtocolError("role cancellation was not acknowledged")
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
        self._session = (
            connection.session
            if connection is not None
            else ClientStageSession(role.weight_index, role.recipient_public)
        )
        self._identity = identity
        self._connection = connection
        self.client: Any = None
        self._connect()
        self._used = False
        self._rotation_interval = rotation_interval
        self._completed = 0
        self._frame_bytes = frame_bytes
        self._call_lock = threading.Lock()
        self._execution_lock = connection.call_lock if connection is not None else self._call_lock
        self._frame_sizes: set[int] = set()
        self.reconnect_count = 0
        self.observations: list[tuple[float, bool]] = []

    @property
    def session(self) -> ClientStageSession:
        return self._connection.session if self._connection is not None else self._session

    def _connect(self) -> None:
        if self._connection is not None:
            self._connection.connect()
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
                    self._session = ClientStageSession(
                        self.role.weight_index, self.role.recipient_public
                    )
                    self._completed = 0
                    self._used = False
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
        request_hash = hashlib.sha256(request_id.encode()).digest()[:16]
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
        sealed = self.session.sender.seal(
            request_id, "u8", [len(operation_payload)], operation_payload
        )
        packet = pad_packet(self.session.wire_packet(sealed), self._frame_bytes)
        if self.role.relay is not None:
            packet = self._quic.seal_onion(
                [
                    (
                        self.role.relay.relay_public_key,
                        f"quic://{self.role.relay.worker_address}",
                    ),
                    (self.role.relay.worker_public_key, ""),
                ],
                secrets.token_bytes(32),
                0,
                packet,
            )[0]
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
        result_dtype, result_shape, payload = self.session.receiver.open_tensor(
            unpad_packet(response, self._frame_bytes)
        )
        if (
            result_dtype != "u8"
            or len(result_shape) != 1
            or result_shape[0] == 0
            or result_shape[0] % 4
            or len(payload) != result_shape[0]
        ):
            raise ProtocolError("role result must contain canonical 32-bit field elements")
        result = [
            int.from_bytes(payload[offset : offset + 4], "little", signed=False)
            for offset in range(0, len(payload), 4)
        ]
        if self._connection is not None:
            self._connection.completed += 1
            completed = self._connection.completed
        else:
            self._completed += 1
            completed = self._completed
        if completed % self._rotation_interval == 0:
            self.session.rotate(completed // self._rotation_interval)
        return result, sent, received

    def close(self) -> None:
        if self._connection is not None:
            return
        with self._execution_lock:
            try:
                if self._used:
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
            if not xurcroh_quic.verify_fortress_product(
                canonical.tolist(),
                product.tolist(),
                share.challenge.r.tolist(),
                share.challenge.v.tolist(),
                prime,
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
        recover_role: Callable[[str], None] | None = None,
    ) -> None:
        if not roles or cover_interval <= 0:
            raise ValueError("a Fortress grid requires roles and a positive cover interval")
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
        while not self._cover_stop.wait(self.cover_interval):
            with self._lock:
                clients = list(self._clients)
            if not clients:
                continue
            client = clients[index % len(clients)]
            index += 1
            try:
                shares = [
                    torch.randint(
                        0,
                        FIELD_MODULUS,
                        (client.operation.quantized.dimensions[1],),
                        dtype=torch.int64,
                    )
                    for _ in range(len(client.operation.shares))
                ]
                client._execute_token(shares, f"cover:{uuid.uuid4().hex}", 0, cover=True)
            except Exception as error:
                client._cover_error = error
                client.cover_failure_count += 1
                self._cover_error = error
                self.cover_failure_count += 1
                return

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
            return combine_role_results(
                activation_shares, self.operation.shares, results, self._providers
            )
        except RoleExecutionError as error:
            with self._lock:
                self._quarantined = self._quarantined or error
            raise

    def _cover_loop(self) -> None:
        features = self.operation.quantized.dimensions[1]
        while not self._cover_stop.wait(self._cover_interval):
            try:
                shares = [
                    torch.randint(0, FIELD_MODULUS, (features,), dtype=torch.int64)
                    for _ in range(len(self.operation.shares))
                ]
                self._execute_token(shares, f"cover:{uuid.uuid4().hex}", 0, cover=True)
            except Exception as error:
                self._cover_error = error
                self.cover_failure_count += 1
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
                raise ProtocolError(
                    "Fortress grid cover traffic failed"
                ) from self._runtime._cover_error
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
        outputs = [
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
            for row in range(rows.shape[0])
        ]
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
