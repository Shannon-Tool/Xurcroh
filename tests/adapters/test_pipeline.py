from __future__ import annotations

import pytest

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError


def test_pipeline_interceptor_round_trip_tamper_and_replay() -> None:
    master = bytes(range(32))
    sender = PipelineTensorInterceptor(master, b"sender", b"receiver", b"pipeline-test")
    receiver = PipelineTensorInterceptor(master, b"receiver", b"sender", b"pipeline-test")
    packet = sender.seal("request-1", "bf16", [2, 2], bytes(range(8)))
    payload = receiver.open(packet)
    assert payload == bytes(range(8))

    with pytest.raises(ProtocolError):
        receiver.open(packet)
    with pytest.raises(ProtocolError):
        receiver.open(packet[:-1] + bytes([packet[-1] ^ 1]))
