from __future__ import annotations

import asyncio
import threading

import xurcroh_quic
from tests.fortress.conftest import MutualTLSIdentities


def test_datagram_relays_carry_real_mutual_tls_quic(
    mtls_identities: MutualTLSIdentities,
) -> None:
    from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay, UdpDatagramRelay

    async def exercise() -> None:
        target = xurcroh_quic.TensorServer.bind_mutual(
            "127.0.0.1:0",
            [mtls_identities["server_certificate"], mtls_identities["ca"]],
            mtls_identities["server_key"],
            mtls_identities["ca"],
        )
        tcp = await TcpDatagramRelay.bind("127.0.0.1", 0, *target.local_addr().rsplit(":", 1))
        udp = await UdpDatagramRelay.bind("127.0.0.1", 0, "127.0.0.1", tcp.port)

        def echo() -> None:
            connection = target.accept()
            values = connection.receive()
            connection.send(*values[:3], values[3], values[4] or "", values[5], values[6])
            connection.close()

        thread = threading.Thread(target=echo, daemon=True)
        thread.start()

        def request() -> bytes:
            client = xurcroh_quic.TensorClient.connect_mutual(
                udp.address,
                "localhost",
                mtls_identities["ca"],
                [mtls_identities["client_certificate"], mtls_identities["ca"]],
                mtls_identities["client_key"],
            )
            client.send(bytes(16), 0, 0, "tensor", "u8", [4], b"test")
            response = client.receive()
            client.close()
            return response[-1]

        assert await asyncio.to_thread(request) == b"test"
        udp.close()
        tcp.close()
        target.close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    asyncio.run(exercise())


def test_tcp_relay_treats_clean_disconnect_as_normal() -> None:
    from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay

    class Writer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def exercise() -> None:
        relay = TcpDatagramRelay.__new__(TcpDatagramRelay)
        relay.target_host = "127.0.0.1"
        relay.target_port = 9
        reader = asyncio.StreamReader()
        reader.feed_eof()
        await relay.handle(reader, Writer())  # type: ignore[arg-type]

    asyncio.run(exercise())
