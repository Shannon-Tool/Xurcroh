from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

from xurcroh.errors import ProtocolError
from xurcroh.fortress.transport import MutualTLSIdentity, RelayHop, RelayRoute

__all__ = ["ObliviousRelay", "RelayHop", "RelayRoute"]


class ObliviousRelay:
    def __init__(
        self,
        onion_private_key: bytes,
        *,
        certificates: tuple[bytes, ...],
        private_key: bytes,
        client_ca: bytes,
        upstream_identity: MutualTLSIdentity,
        allowed_workers: set[str],
        address: str = "127.0.0.1:0",
    ) -> None:
        import xurcroh_quic

        if not allowed_workers:
            raise ValueError("oblivious relay requires an explicit worker allowlist")
        self._quic = xurcroh_quic
        self._onion_private_key = onion_private_key
        self._allowed_workers = allowed_workers
        self._upstream_identity = upstream_identity
        self.server = xurcroh_quic.TensorServer.bind_mutual(
            address, list(certificates), private_key, client_ca
        )
        self.address = self.server.local_addr()
        self.forward_count = 0
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)

    def start(self) -> None:
        if self._closed.is_set():
            raise ProtocolError("oblivious relay is closed")
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

    def _connect_upstream(self, next_hop: str) -> Any:
        if next_hop not in self._allowed_workers or not next_hop.startswith("quic://"):
            raise ProtocolError("onion next hop is not an authorized Fortress worker")
        identity = self._upstream_identity
        return self._quic.TensorClient.connect_mutual(
            next_hop.removeprefix("quic://"),
            identity.server_name,
            identity.pinned_ca,
            list(identity.certificates),
            identity.private_key,
        )

    def _execute(self, connection: Any) -> None:
        upstream: Any = None
        try:
            while not self._closed.is_set():
                request_id, stream_id, sequence, kind, dtype, _, packet = connection.receive()
                if kind == "cancel" and upstream is not None:
                    upstream.send(request_id, stream_id, sequence, kind, "", [], packet)
                    response = upstream.receive()
                    connection.send(
                        *response[:3], response[3], response[4] or "", response[5], response[6]
                    )
                    upstream.close()
                    connection.close()
                    return
                if kind != "tensor" or dtype != "u8":
                    raise ProtocolError("oblivious relay accepts only encrypted tensor frames")
                next_hop, _, _, remaining = self._quic.unwrap_onion(self._onion_private_key, packet)
                if upstream is None:
                    upstream = self._connect_upstream(next_hop)
                upstream.send(
                    request_id, stream_id, sequence, kind, dtype, [len(remaining)], remaining
                )
                response = upstream.receive()
                connection.send(
                    *response[:3], response[3], response[4] or "", response[5], response[6]
                )
                self.forward_count += 1
        except (ValueError, TypeError, ProtocolError) as error:
            if not self._closed.is_set():
                logging.getLogger(__name__).debug(
                    "oblivious relay connection ended: %s", type(error).__name__, exc_info=error
                )
            if upstream is not None:
                with contextlib.suppress(ValueError):
                    upstream.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.server.close()
        if self._accept_thread.is_alive():
            self._accept_thread.join(timeout=10)
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=10)
