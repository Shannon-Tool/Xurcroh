from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from xurcroh import kem
from xurcroh.errors import CompatibilityError, ProtocolError
from xurcroh.fortress.arithmetic import FIELD_MODULUS, _seeded_uniform
from xurcroh.fortress.checkpoint import (
    CHECKPOINT_OPERATION,
    SessionTranscript,
    artifact_set_digest,
    checkpoint_nonce,
    sign_checkpoint,
)
from xurcroh.fortress.native import canonical_field_words
from xurcroh.fortress.onion import unwrap_onion
from xurcroh.fortress.session import (
    HANDSHAKE_MAGIC,
    RoleSessionStore,
    channel_binding,
    split_data_packet,
)
from xurcroh.fortress.transport import (
    DEFAULT_FRAME_BYTES,
    decode_role_artifact,
    native_role_matrix,
    pad_packet,
    unpad_packet,
)
from xurcroh.memory_native import activate_burn_on_read_allocator
from xurcroh.signatures import HybridSigningKey

MAX_ROLE_PACKET_BYTES = 16 * 1024 * 1024
MAX_ACTIVATION_ELEMENTS = 1_048_576
MAX_OPERATION_NAME_BYTES = 1024
SESSIONS_PER_OPERATION = 4


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
        session_capacity: int | None = None,
        rotation_interval: int = 64,
        frame_bytes: int = DEFAULT_FRAME_BYTES,
        onion_private_key: bytes | None = None,
        device: str = "cpu",
        cuda_library: str | Path | None = None,
        strict: bool = True,
        attest: Callable[[int], bool] | None = None,
        allow_unattested: bool = False,
        identity_key: bytes | None = None,
    ) -> None:
        import xurcroh_quic

        if attest is not None and allow_unattested:
            raise ValueError("allow_unattested cannot override a deployment attestation policy")
        if attest is None and not allow_unattested:
            raise ValueError(
                "Fortress role worker requires an attestation callback or "
                "allow_unattested=True for non-TEE development deployments"
            )

        if frame_bytes < 256:
            raise ValueError("fixed frame size is too small")
        self._frame_bytes = frame_bytes
        if device not in {"cpu", "cuda"}:
            raise ValueError("Fortress worker device must be cpu or cuda")
        self.device = device
        if rotation_interval <= 0:
            raise ValueError("rotation interval must be positive")
        self._rotation_interval = rotation_interval
        self._quic = xurcroh_quic
        self._onion_key = (
            None if onion_private_key is None else kem.HybridKemKey.from_bytes(onion_private_key)
        )
        serialized = {"operation": artifacts} if isinstance(artifacts, bytes) else artifacts
        if not serialized:
            raise ValueError("Fortress role worker requires at least one operation")
        if CHECKPOINT_OPERATION in serialized:
            raise ValueError(f"operation name {CHECKPOINT_OPERATION!r} is reserved")
        # Digest the bytes as received, before decoding, so the value names exactly
        # what was deployed to this role.
        self.artifact_digest = artifact_set_digest(serialized)
        self._identity = None if identity_key is None else HybridSigningKey.from_bytes(identity_key)
        self._transcripts: dict[str, SessionTranscript] = {}
        self.artifacts = {
            name: decode_role_artifact(artifact, attest=attest, allow_unattested=allow_unattested)
            for name, artifact in serialized.items()
        }
        weight_indices = {artifact.weight_index for artifact in self.artifacts.values()}
        if len(weight_indices) != 1:
            raise ValueError("all operations in one role must use the same weight-share index")
        self.weight_index = weight_indices.pop()
        recipient = kem.HybridKemKey.from_bytes(recipient_private_key)
        # Every operation holds its own session with a role, so a fixed bound
        # rejects any model with more operations than the bound. The default
        # scales with the assigned operations and leaves headroom for
        # concurrent clients; it stays an explicit, finite limit.
        if session_capacity is None:
            session_capacity = max(64, SESSIONS_PER_OPERATION * len(self.artifacts))
        if session_capacity < len(self.artifacts):
            raise ValueError("session capacity is below the number of assigned operations")
        self.sessions = RoleSessionStore(
            recipient,
            self.weight_index,
            artifact_digest=self.artifact_digest,
            identity=self._identity,
            capacity=session_capacity,
        )
        self._session_completed: dict[str, int] = {}
        self._active_sessions: set[str] = set()
        self._native_weights = (
            {}
            if device == "cuda"
            else {name: native_role_matrix(artifact) for name, artifact in self.artifacts.items()}
        )
        self._cuda_controller: Any = None
        self._cuda_weights: dict[str, torch.Tensor] = {}
        if device == "cuda":
            library = (
                cuda_library
                or os.environ.get("XURCROH_NATIVE_LIBRARY")
                or Path(__file__).resolve().parents[3] / "native/build/xurcroh_burn_alloc.so"
            )
            if not Path(library).is_file():
                raise ValueError("Fortress CUDA worker requires the compiled native library")
            try:
                activate_burn_on_read_allocator(library)
            except CompatibilityError as error:
                if strict:
                    raise
                logging.getLogger(__name__).error(
                    "burn-on-read allocator is not active; CUDA allocations are not zeroed: %s",
                    error,
                )
            from xurcroh.fortress.cuda_driver import (
                collect_driver_diagnostics,
                require_compatible_driver,
            )

            require_compatible_driver(collect_driver_diagnostics())
            if not torch.cuda.is_available():
                raise ValueError("Fortress CUDA worker requires an available CUDA device")
            from xurcroh.fortress.native import FortressNativeController

            self._cuda_controller = FortressNativeController(library)
            self._cuda_weights = {
                name: artifact.matrix.remainder(FIELD_MODULUS).to(torch.int32).T.contiguous().cuda()
                for name, artifact in self.artifacts.items()
            }
        deadline = time.monotonic() + 30
        while True:
            try:
                self.server = xurcroh_quic.TensorServer.bind_mutual(
                    address, list(certificates), private_key, client_ca
                )
                break
            except ValueError as error:
                if (
                    "address already in use" not in str(error).lower()
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.05)
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

    def _expand_activation_seed(self, seed: bytes, features: int) -> torch.Tensor:
        if self._cuda_controller is not None:
            return cast(torch.Tensor, self._cuda_controller.expand_field_device(seed, features))
        return _seeded_uniform(seed, (features,))

    def _decode_activation(self, payload: bytes) -> tuple[str, torch.Tensor]:
        seeded = payload.startswith(b"XF2\x01")
        header = 4 if seeded else 0
        if len(payload) < header + 2:
            raise ProtocolError("Fortress operation header is truncated")
        name_length = int.from_bytes(payload[header : header + 2], "little")
        if name_length == 0 or name_length > MAX_OPERATION_NAME_BYTES:
            raise ProtocolError("Fortress operation name is invalid")
        name_start = header + 2
        name_end = name_start + name_length
        if name_end > len(payload):
            raise ProtocolError("Fortress operation name is truncated")
        try:
            name = payload[name_start:name_end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ProtocolError("Fortress operation name is not ASCII") from error
        if name not in self.artifacts:
            raise ProtocolError("Fortress operation is not assigned to this role")
        input_features = self.artifacts[name].matrix.shape[1]
        activation_payload = payload[name_end:]
        if input_features > MAX_ACTIVATION_ELEMENTS:
            raise ProtocolError("Fortress activation dimensions are invalid")
        if seeded:
            if len(activation_payload) != 32:
                raise ProtocolError("seeded Fortress activation requires a 32-byte seed")
            return name, self._expand_activation_seed(activation_payload, input_features)
        if len(activation_payload) != input_features * 4:
            raise ProtocolError("Fortress activation dimensions are invalid")
        activation = torch.from_numpy(
            np.frombuffer(activation_payload, dtype=np.uint32).astype(np.int64)
        )
        if bool(torch.any((activation < 0) | (activation >= FIELD_MODULUS))):
            raise ProtocolError("Fortress activation must contain canonical field elements")
        return name, activation

    def _execute_activation(self, name: str, activation: torch.Tensor) -> list[int]:
        if self._cuda_controller is None:
            return [
                int(value)
                for value in self._quic.execute_fortress_role(
                    activation.remainder(FIELD_MODULUS).tolist(), self._native_weights[name]
                )
            ]
        if activation.device.type == "cuda":
            if activation.dtype != torch.int32 or not activation.is_contiguous():
                raise ValueError("CUDA activations must be contiguous int32 field elements")
            activation_device = activation
        else:
            activation_device = activation.remainder(FIELD_MODULUS).to(torch.int32).cuda()
        output = torch.empty(self.artifacts[name].matrix.shape[0], dtype=torch.int32, device="cuda")
        try:
            self._cuda_controller.matvec_device(self._cuda_weights[name], activation_device, output)
            return canonical_field_words(output.cpu()).tolist()
        finally:
            activation_device.zero_()
            output.zero_()

    def _checkpoint(self, transcript: SessionTranscript, nonce: bytes) -> bytes:
        if self._identity is None:
            raise ProtocolError("this role has no identity key and cannot sign a checkpoint")
        return sign_checkpoint(
            self._identity,
            transcript,
            artifacts=self.artifact_digest,
            weight_index=self.weight_index,
            nonce=nonce,
        )

    def _execute(self, connection: Any) -> None:
        try:
            while not self._closed.is_set():
                request_id, _, _, kind, dtype, shape, packet = connection.receive()
                if kind == "cancel":
                    session_id = request_id.hex()
                    self.sessions.authorize(session_id, packet)
                    with self._lock:
                        self._requests.pop(session_id, None)
                        self._session_completed.pop(session_id, None)
                        self._transcripts.pop(session_id, None)
                        self._active_sessions.discard(session_id)
                        remaining = bool(self._active_sessions)
                    self.sessions.remove(session_id)
                    connection.send(request_id, self.weight_index, 0, "complete", "", [], b"")
                    if not remaining:
                        connection.close()
                        return
                    continue
                if kind == "tensor" and self._onion_key is not None:
                    next_hop, packet = unwrap_onion(self._onion_key, packet)
                    if next_hop:
                        raise ProtocolError("worker received an onion layer for another hop")
                    shape = [len(packet)]
                if (
                    kind != "tensor"
                    or dtype != "u8"
                    or shape != [self._frame_bytes]
                    or len(packet) != self._frame_bytes
                ):
                    raise ProtocolError("Fortress role expects a fixed-size encrypted frame")
                packet = unpad_packet(packet, self._frame_bytes)
                if packet.startswith(HANDSHAKE_MAGIC):
                    binding = channel_binding(connection, relayed=self._onion_key is not None)
                    session_id, reply = self.sessions.accept(packet, binding)
                    with self._lock:
                        self._active_sessions.add(session_id)
                    reply = pad_packet(reply, self._frame_bytes)
                    connection.send(
                        request_id, self.weight_index, 0, "tensor", "u8", [len(reply)], reply
                    )
                    continue
                session_id, sealed_request = split_data_packet(packet)
                try:
                    receiver, sender = self.sessions.channels(session_id)
                except ProtocolError:
                    # An expired session leaves nothing behind; the client must
                    # start a new one with a fresh handshake.
                    with self._lock:
                        self._requests.pop(session_id, None)
                        self._session_completed.pop(session_id, None)
                        self._transcripts.pop(session_id, None)
                        self._active_sessions.discard(session_id)
                    raise
                state_key = f"{session_id}:{request_id.hex()}"
                with self._lock:
                    self._requests.setdefault(session_id, set()).add(state_key)
                activation = torch.empty(0, dtype=torch.int64)
                exact_payload = bytearray()
                try:
                    result_dtype, result_shape, payload = receiver.open_tensor(sealed_request)
                    if result_dtype != "u8" or result_shape != [len(payload)]:
                        raise ProtocolError("Fortress operation payload must be a byte vector")
                    with self._lock:
                        transcript = self._transcripts.setdefault(
                            session_id, SessionTranscript(session_id)
                        )
                    nonce = checkpoint_nonce(payload)
                    if nonce is not None:
                        exact_payload = bytearray(self._checkpoint(transcript, nonce))
                    else:
                        name, activation = self._decode_activation(payload)
                        values = self._execute_activation(name, activation)
                        exact_payload = bytearray(
                            b"".join(value.to_bytes(4, "little", signed=False) for value in values)
                        )
                    sealed_response = sender.seal(
                        request_id.hex(), "u8", [len(exact_payload)], bytes(exact_payload)
                    )
                    response = pad_packet(sealed_response, self._frame_bytes)
                    connection.send(
                        request_id,
                        self.weight_index,
                        0,
                        "tensor",
                        "u8",
                        [len(response)],
                        response,
                    )
                    transcript.update(sealed_request, sealed_response)
                    with self._lock:
                        completed = self._session_completed.get(session_id, 0) + 1
                        self._session_completed[session_id] = completed
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
        deadline = time.monotonic() + 10
        self.server.close()
        if self._accept_thread.is_alive():
            self._accept_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            threads = list(self._threads)
            self._requests.clear()
            self._session_completed.clear()
            self._transcripts.clear()
            self._active_sessions.clear()
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self.sessions.close()
        for weights in self._cuda_weights.values():
            weights.zero_()
        self._cuda_weights.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for artifact in self.artifacts.values():
            artifact.matrix.zero_()
