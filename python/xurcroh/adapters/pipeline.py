from __future__ import annotations

import hashlib
from typing import cast

from xurcroh.errors import ProtocolError


class PipelineTensorInterceptor:
    def __init__(
        self, master_key: bytes, send_label: bytes, receive_label: bytes, context: bytes
    ) -> None:
        try:
            import xurcroh_quic
        except ImportError as error:
            raise ProtocolError("Rust transport extension is not installed") from error
        self._quic = xurcroh_quic
        self._master_key = master_key
        self._send_label = send_label
        self._receive_label = receive_label
        self._context = context
        self._session = xurcroh_quic.AeadSession(master_key, send_label, receive_label, context)

    def rotate(self, epoch: int) -> None:
        self._session.rotate(
            self._master_key,
            self._send_label,
            self._receive_label,
            self._context,
            epoch,
        )

    def seal(
        self,
        request_id: str,
        dtype: str,
        shape: list[int],
        payload: bytes,
    ) -> bytes:
        request_hash = hashlib.sha256(request_id.encode()).digest()[:16]
        frame = self._quic.encode_frame(
            request_hash,
            0,
            0,
            "tensor",
            dtype,
            shape,
            payload,
        )
        sequence, nonce, ciphertext = self._session.seal(request_hash, frame)
        return cast(bytes, request_hash + sequence.to_bytes(8, "little") + nonce + ciphertext)

    def open_tensor(self, packet: bytes) -> tuple[str, list[int], bytes]:
        if len(packet) < 16 + 8 + 12 + 16:
            raise ProtocolError("truncated pipeline tensor packet")
        request_hash = packet[:16]
        sequence = int.from_bytes(packet[16:24], "little")
        nonce = packet[24:36]
        ciphertext = packet[36:]
        try:
            frame = self._session.open(request_hash, sequence, nonce, ciphertext)
            _, _, _, _, dtype, shape, payload = self._quic.decode_frame(frame)
            if dtype is None:
                raise ProtocolError("pipeline frame has no tensor descriptor")
            return dtype, shape, cast(bytes, payload)
        except ValueError as error:
            raise ProtocolError(str(error)) from error

    def open(self, packet: bytes) -> bytes:
        return self.open_tensor(packet)[2]
