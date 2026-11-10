from __future__ import annotations

import pytest


def test_rust_frame_and_session_python_bindings() -> None:
    xurcroh_quic = pytest.importorskip("xurcroh_quic")
    request_id = bytes(range(16))
    encoded = xurcroh_quic.encode_frame(
        request_id,
        2,
        9,
        "tensor",
        "bf16",
        [2, 2],
        bytes(range(8)),
    )
    decoded = xurcroh_quic.decode_frame(encoded)
    assert decoded == (request_id, 2, 9, "tensor", "bf16", [2, 2], bytes(range(8)))
    with pytest.raises(ValueError):
        xurcroh_quic.decode_frame(encoded[:-1])

    client = xurcroh_quic.AeadSession(bytes(range(32)), b"client", b"server", b"test")
    server = xurcroh_quic.AeadSession(bytes(range(32)), b"server", b"client", b"test")
    sequence, nonce, ciphertext = client.seal(b"header", b"tensor")
    assert sequence == 0
    assert server.open(b"header", sequence, nonce, ciphertext) == b"tensor"
    with pytest.raises(ValueError, match="replay"):
        server.open(b"header", sequence, nonce, ciphertext)


def test_native_quic_client_server_response() -> None:
    import threading

    xurcroh_quic = pytest.importorskip("xurcroh_quic")
    certificate, key = xurcroh_quic.generate_self_signed("localhost")
    server = xurcroh_quic.TensorServer("127.0.0.1:0", certificate, key)
    address = server.local_addr()
    failures = []

    def respond() -> None:
        try:
            connection = server.accept()
            request_id, stream_id, sequence, kind, dtype, shape, payload = connection.receive()
            connection.send(request_id, stream_id, sequence, kind, dtype, shape, payload)
            connection.close()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=respond, daemon=True)
    thread.start()
    client = xurcroh_quic.TensorClient(address, "localhost", certificate)
    request_id = bytes(range(16))
    client.send(request_id, 1, 4, "tensor", "u8", [4], b"test")
    response = client.receive()
    client.close()
    thread.join(timeout=10)
    server.close()
    assert not failures
    assert response == (request_id, 1, 4, "tensor", "u8", [4], b"test")
