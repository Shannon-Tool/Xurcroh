from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from xurcroh.stage_worker import DenseStage, _decode_tensor, _encode_tensor


class NativeStageServer:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        stage_index: int,
        master_key: bytes,
        certificate: bytes,
        private_key: bytes,
        address: str = "127.0.0.1:0",
    ) -> None:
        import xurcroh_quic

        self._quic = xurcroh_quic
        self.stage = DenseStage(manifest_path, stage_index, master_key)
        self.server = xurcroh_quic.TensorServer(address, certificate, private_key)
        self.address = self.server.local_addr()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
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
            request_id, _, _, kind, dtype, shape, packet = connection.receive()
            if kind == "cancel":
                self.stage.runner.drop_request(request_id.hex())
                connection.close()
                return
            if kind != "tensor" or dtype != "u8" or shape != [len(packet)]:
                raise ValueError("native stage expects a bounded byte tensor")
            hidden = _decode_tensor(packet, self.stage.receiver)
            output = self.stage.execute(hidden, request_id.hex())
            response, _ = _encode_tensor(output, request_id.hex(), self.stage.sender)
            connection.send(
                request_id,
                self.stage.stage_index,
                0,
                "tensor",
                "u8",
                [len(response)],
                response,
            )
            connection.close()
            hidden.zero_()
            output.zero_()
        except (ValueError, TypeError) as error:
            logging.getLogger(__name__).warning(
                "native stage request failed: %s", type(error).__name__
            )

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.server.close()
        self._accept_thread.join(timeout=10)
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=10)
        self.stage.close()
