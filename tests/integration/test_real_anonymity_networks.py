"""Runs against real Tor and I2P daemons. Skipped unless the environment names them.

Tor: run ``tor`` with an onion service whose port 9446 forwards to
``127.0.0.1:$XURCROH_CARRIER_RELAY_PORT``, then set ``XURCROH_TOR_SOCKS=HOST:PORT`` and
``XURCROH_TOR_ONION=<address>.onion``.

I2P: run ``i2pd`` with the SAM bridge enabled and a server tunnel to the same relay port,
then set ``XURCROH_I2P_SAM=HOST:PORT`` and ``XURCROH_I2P_DESTINATION=<address>.b32.i2p``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest
import xurcroh_quic
from tests.fortress.conftest import MutualTLSIdentities, mtls_identities  # noqa: F401

from xurcroh.fortress.carriers import CarrierRelay, CarrierSpec
from xurcroh.fortress.datagram_tunnel import TcpDatagramRelay

ROUND_TRIPS = 8


def _measure(kind: str, target: str, proxy: str, mtls: MutualTLSIdentities) -> dict[str, object]:
    relay_port = int(os.environ["XURCROH_CARRIER_RELAY_PORT"])
    proxy_host, proxy_port = proxy.rsplit(":", 1)

    async def exercise() -> dict[str, object]:
        server = xurcroh_quic.TensorServer.bind_mutual(
            "127.0.0.1:0",
            [mtls["server_certificate"], mtls["ca"]],
            mtls["server_key"],
            mtls["ca"],
        )
        tcp = await TcpDatagramRelay.bind(
            "127.0.0.1", relay_port, *server.local_addr().rsplit(":", 1)
        )
        relay = await asyncio.to_thread(
            CarrierRelay, CarrierSpec(kind, target, 9446, proxy_host, int(proxy_port))
        )

        def echo() -> None:
            connection = server.accept()
            for _ in range(ROUND_TRIPS):
                values = connection.receive()
                connection.send(*values[:3], values[3], values[4] or "", values[5], values[6])
            connection.close()

        thread = threading.Thread(target=echo, daemon=True)
        thread.start()

        def request() -> dict[str, object]:
            started = time.perf_counter()
            client = xurcroh_quic.TensorClient.connect_mutual(
                relay.address,
                "localhost",
                mtls["ca"],
                [mtls["client_certificate"], mtls["ca"]],
                mtls["client_key"],
            )
            connected = time.perf_counter() - started
            group = client.key_exchange_group()
            frame = bytes(16_384)
            timings = []
            for sequence in range(ROUND_TRIPS):
                began = time.perf_counter()
                client.send(bytes(16), 0, sequence, "tensor", "u8", [len(frame)], frame)
                assert bytes(client.receive()[-1]) == frame
                timings.append(time.perf_counter() - began)
            client.close()
            return {
                "carrier": kind,
                "key_exchange_group": group,
                "connect_seconds": round(connected, 3),
                "round_trip_seconds": [round(value, 3) for value in timings],
                "frame_bytes": len(frame),
            }

        try:
            return await asyncio.to_thread(request)
        finally:
            failures = relay.failures
            await asyncio.to_thread(relay.close)
            tcp.close()
            server.close()
            thread.join(timeout=5)
            if failures:
                print("carrier failures:", failures)

    report = asyncio.run(exercise())
    output = os.environ.get("XURCROH_CARRIER_REPORT")
    if output:
        path = Path(output)
        existing = json.loads(path.read_text()) if path.is_file() else []
        path.write_text(json.dumps([*existing, report], indent=2) + "\n")
    return report


@pytest.mark.skipif(
    not (os.environ.get("XURCROH_TOR_SOCKS") and os.environ.get("XURCROH_TOR_ONION")),
    reason="needs a running Tor daemon and onion service (see module docstring)",
)
def test_real_quic_through_real_tor_onion_service(
    mtls_identities: MutualTLSIdentities,  # noqa: F811
) -> None:
    report = _measure(
        "tor",
        os.environ["XURCROH_TOR_ONION"],
        os.environ["XURCROH_TOR_SOCKS"],
        mtls_identities,
    )

    assert report["key_exchange_group"] == "X25519MLKEM768"
    assert len(report["round_trip_seconds"]) == ROUND_TRIPS  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (os.environ.get("XURCROH_I2P_SAM") and os.environ.get("XURCROH_I2P_DESTINATION")),
    reason="needs a running i2pd with SAM and a server tunnel (see module docstring)",
)
def test_real_quic_through_real_i2p(
    mtls_identities: MutualTLSIdentities,  # noqa: F811
) -> None:
    report = _measure(
        "i2p",
        os.environ["XURCROH_I2P_DESTINATION"],
        os.environ["XURCROH_I2P_SAM"],
        mtls_identities,
    )

    assert report["key_exchange_group"] == "X25519MLKEM768"
    assert len(report["round_trip_seconds"]) == ROUND_TRIPS  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (os.environ.get("XURCROH_TOR_SOCKS") and os.environ.get("XURCROH_TOR_ONION")),
    reason="needs a running Tor daemon and onion service (see module docstring)",
)
def test_complete_model_request_through_real_tor(
    tiny_llama_path: Path,
    tmp_path: Path,
    mtls_identities: MutualTLSIdentities,  # noqa: F811
) -> None:
    """A whole chat request with every role worker reached through its own Tor circuit.

    The onion service must also forward ports 9447-9450 to the four ports after
    ``XURCROH_CARRIER_RELAY_PORT``.
    """

    from tests.fortress.test_sdk import _signed_remote_grid

    from Xurcroh import Xurcroh
    from xurcroh.receipts import verify_trace

    base = int(os.environ["XURCROH_CARRIER_RELAY_PORT"])
    cluster, workers = _signed_remote_grid(tiny_llama_path, tmp_path, mtls_identities, 2)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    relays = []
    try:
        for index, worker in enumerate(cluster):
            host, port = worker["endpoint"].removeprefix("quic://").rsplit(":", 1)
            relays.append(
                asyncio.run_coroutine_threadsafe(
                    TcpDatagramRelay.bind("127.0.0.1", base + 1 + index, host, int(port)), loop
                ).result(timeout=30)
            )
            worker["labels"].update(
                {
                    "carrier": "tor",
                    "carrier_target": f"{os.environ['XURCROH_TOR_ONION']}:{9447 + index}",
                    "carrier_proxy": os.environ["XURCROH_TOR_SOCKS"],
                    "server_name": "localhost",
                }
            )
        started = time.perf_counter()
        with Xurcroh(
            cluster=cluster,
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "revision": "a" * 40,
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_shares": 2,
                }
            },
        ) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
                timeout=3000,
            )
            elapsed = time.perf_counter() - started
            trace = client.tasks.trace(response.execution.request_id)
    finally:
        for relay in relays:
            loop.call_soon_threadsafe(relay.close)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        for worker in workers:
            worker.close()

    assert response.usage.completion_tokens == 1
    assert verify_trace(trace)["worker_checkpoints"] == 4
    routes = trace["transport"]["routes"]
    assert [route["carrier"] for route in routes] == ["tor"] * 4
    assert all(route["negotiated"]["key_exchange_group"] == "X25519MLKEM768" for route in routes)
    report = {
        "carrier": "tor",
        "test": "complete model request",
        "model": "tiny Llama test checkpoint, 2x2 role grid",
        "completion_tokens": 1,
        "seconds_including_model_load": round(elapsed, 1),
        "role_requests": trace["transport"]["counters"]["fortress_role_requests"],
        "reconnections": trace["transport"]["counters"]["fortress_reconnections"],
    }
    output = os.environ.get("XURCROH_CARRIER_REPORT")
    if output:
        path = Path(output)
        existing = json.loads(path.read_text()) if path.is_file() else []
        path.write_text(json.dumps([*existing, report], indent=2) + "\n")
