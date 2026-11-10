from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from xurcroh.stage_worker import DenseStage, _decode_tensor, _encode_tensor


class NativeStageServer:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        stage_index: int,
        stage_private_key: X25519PrivateKey,
        certificate: bytes,
        private_key: bytes,
        address: str = "127.0.0.1:0",
    ) -> None:
        import xurcroh_quic

        self._quic = xurcroh_quic
        self.stage = DenseStage(manifest_path, stage_index, stage_private_key)
        self.server = xurcroh_quic.TensorServer(address, certificate, private_key)
        self.address = self.server.local_addr()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._session_requests: dict[str, set[str]] = {}
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)

    def start(self) -> None:
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

    def _execute(self, connection: Any) -> None:
        try:
            while not self._closed.is_set():
                request_id, _, _, kind, dtype, shape, packet = connection.receive()
                if kind == "cancel":
                    session_id = request_id.hex()
                    with self._lock:
                        request_ids = self._session_requests.pop(session_id, set())
                    for cached_request in request_ids:
                        self.stage.runner.drop_request(cached_request)
                    self.stage.sessions.remove(session_id)
                    connection.send(request_id, self.stage.stage_index, 0, "complete", "", [], b"")
                    return
                if kind != "tensor" or dtype != "u8" or shape != [len(packet)]:
                    raise ValueError("native stage expects a bounded byte tensor")
                if len(packet) < 65:
                    raise ValueError("native stage session envelope is truncated")
                client_public = base64.b64encode(packet[:32]).decode()
                session_id = packet[32:64].decode()
                receiver, sender = self.stage.sessions.get_or_create(session_id, client_public)
                with self._lock:
                    self._session_requests.setdefault(session_id, set()).add(request_id.hex())
                hidden = _decode_tensor(packet[64:], receiver)
                output = self.stage.execute(hidden, request_id.hex())
                response, _ = _encode_tensor(output, request_id.hex(), sender)
                connection.send(
                    request_id,
                    self.stage.stage_index,
                    0,
                    "tensor",
                    "u8",
                    [len(response)],
                    response,
                )
                hidden.zero_()
                output.zero_()
        except (ValueError, TypeError) as error:
            if not self._closed.is_set():
                logging.getLogger(__name__).debug(
                    "native stage connection ended: %s", type(error).__name__
                )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.server.close()
        self._accept_thread.join(timeout=10)
        with self._lock:
            threads = list(self._threads)
            self._session_requests.clear()
        for thread in threads:
            thread.join(timeout=10)
        self.stage.close()
