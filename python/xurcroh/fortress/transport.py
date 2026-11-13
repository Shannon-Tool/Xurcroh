from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import PRIME_MODULUS, RING_MODULUS
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
    if weight_index not in operation.shares or weight_index not in operation.storage:
        raise ValueError("weight share is not present in the compiled operation")
    if len(recipient_public) != 32:
        raise ValueError("recipient public key must contain 32 bytes")
    share = operation.shares[weight_index]
    stored = operation.storage[weight_index]
    bundle = stored.bundle
    document = {
        "format": "xurcroh-fortress-role-v1",
        "weight_index": weight_index,
        "recipient_public": _encode_bytes(recipient_public),
        "provider_id": share.provider_id,
        "replica_provider_id": share.replica_provider_id,
        "shape": list(stored.shape),
        "storage": {
            "threshold": bundle.threshold,
            "nonce": _encode_bytes(bundle.nonce),
            "aad": _encode_bytes(bundle.aad),
            "slices": {str(index): _encode_bytes(value) for index, value in bundle.slices.items()},
            "key_shares": {
                str(index): _encode_bytes(value) for index, value in bundle.key_shares.items()
            },
        },
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def decode_role_artifact(encoded: bytes) -> RoleArtifact:
    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("role artifact is not valid JSON") from error
    if not isinstance(document, dict) or document.get("format") != "xurcroh-fortress-role-v1":
        raise ProtocolError("unsupported Fortress role artifact")
    try:
        weight_index = int(document["weight_index"])
        shape = tuple(int(value) for value in document["shape"])
        recipient_public = _decode_bytes(document["recipient_public"])
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
    if len(shape) != 2 or any(value <= 0 for value in shape) or len(recipient_public) != 32:
        raise ProtocolError("role artifact dimensions or recipient are invalid")
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


class _RoleTransport:
    def __init__(
        self,
        role: RoleEndpoint,
        identity: MutualTLSIdentity,
        operation_name: str,
        rotation_interval: int,
        frame_bytes: int,
    ) -> None:
        import xurcroh_quic

        self.role = role
        self._quic = xurcroh_quic
        self._operation_name = operation_name.encode("ascii")
        if not self._operation_name or len(self._operation_name) > 1024:
            raise ValueError("operation name must contain 1-1024 ASCII bytes")
        self.session = ClientStageSession(role.weight_index, role.recipient_public)
        self.client = xurcroh_quic.TensorClient.connect_mutual(
            role.relay.relay_address if role.relay is not None else role.address,
            identity.server_name,
            identity.pinned_ca,
            list(identity.certificates),
            identity.private_key,
        )
        self._used = False
        self._rotation_interval = rotation_interval
        self._completed = 0
        self._frame_bytes = frame_bytes
        self._call_lock = threading.Lock()
        self._frame_sizes: set[int] = set()
        self.observations: list[tuple[float, bool]] = []

    def execute(
        self,
        activation: torch.Tensor,
        request_id: str,
        sequence: int,
        *,
        cover: bool = False,
    ) -> tuple[list[int], int, int]:
        started = time.perf_counter()
        with self._call_lock:
            result = self._execute(activation, request_id, sequence)
        self.observations.append((time.perf_counter() - started, cover))
        return result

    def _execute(
        self, activation: torch.Tensor, request_id: str, sequence: int
    ) -> tuple[list[int], int, int]:
        request_hash = hashlib.sha256(request_id.encode()).digest()[:16]
        canonical = activation.detach().cpu().to(torch.int64).remainder(RING_MODULUS).contiguous()
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
            or result_shape[0] % 16
            or len(payload) != result_shape[0]
        ):
            raise ProtocolError("role result must contain exact 128-bit products")
        result = [
            int.from_bytes(payload[offset : offset + 16], "little", signed=True)
            for offset in range(0, len(payload), 16)
        ]
        self._completed += 1
        if self._completed % self._rotation_interval == 0:
            self.session.rotate(self._completed // self._rotation_interval)
        return result, sent, received

    def close(self) -> None:
        try:
            if self._used:
                session_id = bytes.fromhex(self.session.session_id)
                self.client.send(session_id, 0, 0, "cancel", "", [], self.session.token_bytes)
                response_id, _, _, kind, _, _, _ = self.client.receive()
                if response_id != session_id or kind != "complete":
                    raise ProtocolError("role cancellation was not acknowledged")
        finally:
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
        canonical = activation.remainder(RING_MODULUS)
        for weight_index, share in weight_shares.items():
            key = (activation_index, weight_index)
            exact_product = results[key]
            prime = share.challenge.prime
            product = torch.tensor([value % prime for value in exact_product], dtype=torch.int64)
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
    columns: list[int] = []
    for column in range(output_size):
        total = sum(
            results[(activation_index, weight_index)][column]
            for activation_index in range(len(activation_shares))
            for weight_index in weight_shares
        )
        columns.append(total % RING_MODULUS)
    return torch.tensor(columns, dtype=torch.int64)


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
        canonical = activation.remainder(RING_MODULUS).tolist()
        for weight_index, share in weight_shares.items():
            key = (activation_index, weight_index)
            results[key] = xurcroh_quic.execute_fortress_role(
                canonical, native_compiled_weight_share(share)
            )
            providers[key] = share.provider_id
    return combine_role_results(activation_shares, weight_shares, results, providers)


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
        self._executor = ThreadPoolExecutor(max_workers=len(roles))
        self._transports = {
            key: _RoleTransport(role, identity, operation_name, rotation_interval, frame_bytes)
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
        self._cover_thread = threading.Thread(target=self._cover_loop, daemon=True)
        self._cover_thread.start()

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
    ) -> torch.Tensor:
        futures = {
            key: self._executor.submit(
                self._transports[key].execute,
                activation_shares[key[0]],
                request_id,
                sequence,
                cover=cover,
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
        return combine_role_results(
            activation_shares, self.operation.shares, results, self._providers
        )

    def _cover_loop(self) -> None:
        features = self.operation.quantized.values.shape[1]
        while not self._cover_stop.wait(self._cover_interval):
            try:
                shares = [
                    torch.randint(0, 2**32, (features,), dtype=torch.int64)
                    for _ in range(len(self.operation.shares))
                ]
                self._execute_token(shares, f"cover:{uuid.uuid4().hex}", 0, cover=True)
            except Exception as error:
                self._cover_error = error
                return

    def execute(
        self, activation_shares: list[torch.Tensor], *, request_id: str | None = None
    ) -> torch.Tensor:
        with self._lock:
            if self._closed:
                raise ProtocolError("Fortress role client is closed")
            if self._cover_error is not None:
                raise ProtocolError("Fortress cover traffic failed") from self._cover_error
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
            self._execute_token([share[row] for share in rows_by_share], correlation_id, row)
            for row in range(rows.shape[0])
        ]
        return outputs[0] if vector_input else torch.stack(outputs)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._cover_stop.set()
        self._cover_thread.join(timeout=30)
        self._executor.shutdown(wait=True)
        first_error: Exception | None = self._cover_error
        for transport in self._transports.values():
            try:
                transport.close()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise ProtocolError("one or more Fortress roles did not close cleanly") from first_error
