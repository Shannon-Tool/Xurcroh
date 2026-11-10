from __future__ import annotations

import os

from xurcroh.memory_host import SecureMemfd


def test_sealed_memfd_round_trip_and_close() -> None:
    secure = SecureMemfd(b"encrypted payload")
    assert secure.read() == b"encrypted payload"
    secure.close()
    assert secure.closed
    with __import__("pytest").raises(RuntimeError):
        secure.read()
    assert not os.path.exists(f"/proc/self/fd/{secure.fd}")
