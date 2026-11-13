from __future__ import annotations

import base64
import logging
import threading
from collections.abc import Mapping
from typing import Any

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import RING_MODULUS
from xurcroh.fortress.transport import decode_role_artifact, native_role_matrix
from xurcroh.runtime.stage_sessions import SESSION_ENVELOPE_BYTES, StageSessionStore

MAX_ROLE_PACKET_BYTES = 16 * 1024 * 1024
MAX_ACTIVATION_ELEMENTS = 1_048_576
MAX_OPERATION_NAME_BYTES = 1024


class FortressRoleWorker:
    def __init__(
        self,
        artifacts: bytes | Mapping[str, bytes],
        recipient_private_key: bytes,
        *,
        certificates: tuple[bytes, ...],
        private_key: bytes,
        client_ca: bytes,
        address: str = "127.0.0.1:0",
        session_capacity: int = 64,
        rotation_interval: int = 64,
    ) -> None:
        import xurcroh_quic

        if rotation_interval <= 0:
            raise ValueError("rotation interval must be positive")
        self._rotation_interval = rotation_interval
        self._quic = xurcroh_quic
        serialized = {"operation": artifacts} if isinstance(artifacts, bytes) else artifacts
        if not serialized:
            raise ValueError("Fortress role worker requires at least one operation")
        self.artifacts = {
            name: decode_role_artifact(artifact) for name, artifact in serialized.items()
        }
        weight_indices = {artifact.weight_index for artifact in self.artifacts.values()}
        if len(weight_indices) != 1:
            raise ValueError("all operations in one role must use the same weight-share index")
        self.weight_index = weight_indices.pop()
        private = X25519PrivateKey.from_private_bytes(recipient_private_key)
        self.sessions = StageSessionStore(private, self.weight_index, capacity=session_capacity)
        self._native_weights = {
            name: native_role_matrix(artifact) for name, artifact in self.artifacts.items()
        }
        self.server = xurcroh_quic.TensorServer.bind_mutual(
            address, list(certificates), private_key, client_ca
        )
        self.address = self.server.local_addr()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._requests: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)

    @property
    def active_sessions(self) -> int:
        return self.sessions.active_count

    @property
    def active_requests(self) -> int:
        with self._lock:
            return sum(len(requests) for requests in self._requests.values())

    def start(self) -> None:
        if self._closed.is_set():
            raise ProtocolError("Fortress role worker is closed")
        if not self._accept_thread.is_alive():
            self._accept_thread.start()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                connection = self.server.accept()
            except ValueError:
                if self._closed.is_set():
                    return
                continue
            thread = threading.Thread(target=self._execute, args=(connection,), daemon=True)
            with self._lock:
                self._threads.append(thread)
            thread.start()

    def _decode_activation(self, payload: bytes) -> tuple[str, torch.Tensor]:
        if len(payload) < 2:
            raise ProtocolError("Fortress operation header is truncated")
        name_length = int.from_bytes(payload[:2], "little")
        if name_length == 0 or name_length > MAX_OPERATION_NAME_BYTES:
            raise ProtocolError("Fortress operation name is invalid")
        name_end = 2 + name_length
        if name_end > len(payload):
            raise ProtocolError("Fortress operation name is truncated")
        try:
            name = payload[2:name_end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ProtocolError("Fortress operation name is not ASCII") from error
        if name not in self.artifacts:
            raise ProtocolError("Fortress operation is not assigned to this role")
        input_features = self.artifacts[name].matrix.shape[1]
        activation_payload = payload[name_end:]
        if (
            input_features > MAX_ACTIVATION_ELEMENTS
            or len(activation_payload) != input_features * 8
        ):
            raise ProtocolError("Fortress activation dimensions are invalid")
        activation = torch.frombuffer(bytearray(activation_payload), dtype=torch.int64).clone()
        return name, activation

    def _execute(self, connection: Any) -> None:
        completed = 0
        try:
            while not self._closed.is_set():
                request_id, _, _, kind, dtype, shape, packet = connection.receive()
                if kind == "cancel":
                    session_id = request_id.hex()
                    self.sessions.authorize(session_id, packet)
                    with self._lock:
                        self._requests.pop(session_id, None)
                    self.sessions.remove(session_id)
                    connection.send(request_id, self.weight_index, 0, "complete", "", [], b"")
                    connection.close()
                    return
                if (
                    kind != "tensor"
                    or dtype != "u8"
                    or shape != [len(packet)]
                    or len(packet) > MAX_ROLE_PACKET_BYTES
                    or len(packet) <= SESSION_ENVELOPE_BYTES
                ):
                    raise ProtocolError("Fortress role expects a bounded encrypted byte tensor")
                client_public = base64.b64encode(packet[:32]).decode()
                session_id = packet[32:64].decode()
                session_token = packet[64:96]
                handshake = packet[96:SESSION_ENVELOPE_BYTES]
                receiver, sender = self.sessions.get_or_create(
                    session_id, client_public, session_token, handshake
                )
                state_key = f"{session_id}:{request_id.hex()}"
                with self._lock:
                    self._requests.setdefault(session_id, set()).add(state_key)
                activation = torch.empty(0, dtype=torch.int64)
                exact_payload = bytearray()
                try:
                    result_dtype, result_shape, payload = receiver.open_tensor(
                        packet[SESSION_ENVELOPE_BYTES:]
                    )
                    if result_dtype != "u8" or result_shape != [len(payload)]:
                        raise ProtocolError("Fortress operation payload must be a byte vector")
                    name, activation = self._decode_activation(payload)
                    values = self._quic.execute_fortress_role(
                        activation.remainder(RING_MODULUS).tolist(), self._native_weights[name]
                    )
                    exact_payload = bytearray(
                        b"".join(value.to_bytes(16, "little", signed=True) for value in values)
                    )
                    response = sender.seal(
                        request_id.hex(), "u8", [len(exact_payload)], bytes(exact_payload)
                    )
                    connection.send(
                        request_id,
                        self.weight_index,
                        0,
                        "tensor",
                        "u8",
                        [len(response)],
                        response,
                    )
                    completed += 1
                    if completed % self._rotation_interval == 0:
                        self.sessions.rotate(session_id, completed // self._rotation_interval)
                finally:
                    activation.zero_()
                    for index in range(len(exact_payload)):
                        exact_payload[index] = 0
                    with self._lock:
                        requests = self._requests.get(session_id)
                        if requests is not None:
                            requests.discard(state_key)
                            if not requests:
                                self._requests.pop(session_id, None)
        except (ValueError, TypeError, OverflowError, ProtocolError) as error:
            if not self._closed.is_set():
                logging.getLogger(__name__).debug(
                    "Fortress role connection ended: %s", type(error).__name__, exc_info=error
                )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.server.close()
        if self._accept_thread.is_alive():
            self._accept_thread.join(timeout=10)
        with self._lock:
            threads = list(self._threads)
            self._requests.clear()
        for thread in threads:
            thread.join(timeout=10)
        self.sessions.close()
        for artifact in self.artifacts.values():
            artifact.matrix.zero_()
