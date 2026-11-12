from __future__ import annotations

import base64
import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import PRIME_MODULUS, RING_MODULUS, verify_freivalds
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
class RoleEndpoint:
    activation_index: int
    weight_index: int
    address: str
    recipient_public: bytes
    provider_id: str


@dataclass(frozen=True)
class RoleArtifact:
    weight_index: int
    recipient_public: bytes
    matrix: torch.Tensor
    provider_id: str
    replica_provider_id: str | None


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


class _RoleTransport:
    def __init__(
        self,
        role: RoleEndpoint,
        identity: MutualTLSIdentity,
        operation_name: str,
        rotation_interval: int,
    ) -> None:
        import xurcroh_quic

        self.role = role
        self._quic = xurcroh_quic
        self._operation_name = operation_name.encode("ascii")
        if not self._operation_name or len(self._operation_name) > 1024:
            raise ValueError("operation name must contain 1-1024 ASCII bytes")
        self.session = ClientStageSession(role.weight_index, role.recipient_public)
        self.client = xurcroh_quic.TensorClient.connect_mutual(
            role.address,
            identity.server_name,
            identity.pinned_ca,
            list(identity.certificates),
            identity.private_key,
        )
        self._used = False
        self._rotation_interval = rotation_interval
        self._completed = 0

    def execute(
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
        packet = self.session.wire_packet(sealed)
        sent = len(
            self._quic.encode_frame(
                request_hash, 0, sequence, "tensor", "u8", [len(packet)], packet
            )
        )
        self.client.send(request_hash, 0, sequence, "tensor", "u8", [len(packet)], packet)
        self._used = True
        response_id, _, _, kind, dtype, shape, response = self.client.receive()
        received = len(
            self._quic.encode_frame(response_id, 0, sequence, kind, dtype or "", shape, response)
        )
        if response_id != request_hash or kind != "tensor" or dtype != "u8":
            raise ProtocolError(f"role {self.role.provider_id} returned an invalid response")
        if shape != [len(response)]:
            raise ProtocolError("role response tensor descriptor is invalid")
        result_dtype, result_shape, payload = self.session.receiver.open_tensor(response)
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
    columns: list[int] = []
    for column in range(output_size):
        total = 0
        for activation_index, activation in enumerate(activation_shares):
            canonical = activation.remainder(RING_MODULUS)
            for weight_index, share in weight_shares.items():
                key = (activation_index, weight_index)
                exact_product = results[key]
                prime = share.challenge.prime
                product = torch.tensor(
                    [value % prime for value in exact_product], dtype=torch.int64
                )
                if not verify_freivalds(canonical, product, share.challenge):
                    raise RoleExecutionError(
                        activation_index,
                        weight_index,
                        providers[key],
                        "Freivalds check failed",
                    )
                total += exact_product[column]
        columns.append(total % RING_MODULUS)
    return torch.tensor(columns, dtype=torch.int64)


class FortressRoleClient:
    def __init__(
        self,
        operation: CompiledOperation,
        roles: list[RoleEndpoint],
        identity: MutualTLSIdentity,
        *,
        operation_name: str = "operation",
        rotation_interval: int = 64,
    ) -> None:
        if rotation_interval <= 0:
            raise ValueError("rotation interval must be positive")
        self.operation = operation
        self.roles = {(role.activation_index, role.weight_index): role for role in roles}
        expected = {
            (activation_index, weight_index)
            for activation_index in range(2)
            for weight_index in operation.shares
        }
        if len(roles) != len(self.roles) or set(self.roles) != expected:
            raise ValueError("each Fortress grid role requires exactly one endpoint")
        self._providers = {key: role.provider_id for key, role in self.roles.items()}
        self._executor = ThreadPoolExecutor(max_workers=len(roles))
        self._transports = {
            key: _RoleTransport(role, identity, operation_name, rotation_interval)
            for key, role in self.roles.items()
        }
        self._lock = threading.Lock()
        self._closed = False
        self.request_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    def _execute_token(
        self, activation_shares: list[torch.Tensor], request_id: str, sequence: int
    ) -> torch.Tensor:
        futures = {
            key: self._executor.submit(
                self._transports[key].execute,
                activation_shares[key[0]],
                request_id,
                sequence,
            )
            for key in self._transports
        }
        results: dict[tuple[int, int], list[int]] = {}
        for key, future in futures.items():
            result, sent, received = future.result()
            results[key] = result
            with self._lock:
                self.request_count += 1
                self.bytes_sent += sent
                self.bytes_received += received
        return combine_role_results(
            activation_shares, self.operation.shares, results, self._providers
        )

    def execute(
        self, activation_shares: list[torch.Tensor], *, request_id: str | None = None
    ) -> torch.Tensor:
        with self._lock:
            if self._closed:
                raise ProtocolError("Fortress role client is closed")
        if len(activation_shares) != 2:
            raise ValueError("the Fortress grid requires two activation shares")
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
        self._executor.shutdown(wait=True)
        first_error: Exception | None = None
        for transport in self._transports.values():
            try:
                transport.close()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise ProtocolError("one or more Fortress roles did not close cleanly") from first_error
