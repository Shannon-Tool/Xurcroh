from __future__ import annotations

import threading

import xurcroh_quic
from tests.fortress.conftest import MutualTLSIdentities


def test_python_binding_exposes_mutual_tls(mtls_identities: MutualTLSIdentities) -> None:
    server = xurcroh_quic.TensorServer.bind_mutual(
        "127.0.0.1:0",
        [mtls_identities["server_certificate"], mtls_identities["ca"]],
        mtls_identities["server_key"],
        mtls_identities["ca"],
    )
    received: list[bytes] = []

    def echo() -> None:
        connection = server.accept()
        request_id, stream_id, sequence, kind, dtype, shape, payload = connection.receive()
        received.append(payload)
        connection.send(request_id, stream_id, sequence, kind, dtype or "", shape, payload)
        connection.close()

    thread = threading.Thread(target=echo, daemon=True)
    thread.start()
    client = xurcroh_quic.TensorClient.connect_mutual(
        server.local_addr(),
        "localhost",
        mtls_identities["ca"],
        [mtls_identities["client_certificate"], mtls_identities["ca"]],
        mtls_identities["client_key"],
    )
    client.send(bytes(16), 0, 0, "tensor", "u8", [4], b"test")
    response = client.receive()
    client.close()
    thread.join(timeout=5)
    server.close()
    assert response[-1] == b"test"
    assert received == [b"test"]
