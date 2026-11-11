from __future__ import annotations

import ctypes

import pytest

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.memory import lock_buffer


class LibC:
    def __init__(self, status: int) -> None:
        self.mlock = lambda address, length: status


def test_mlock_is_best_effort_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XURCROH_REQUIRE_MLOCK", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC(1))
    assert lock_buffer(bytearray(16)) is False
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC(0))
    assert lock_buffer(bytearray(16)) is True


def test_required_mlock_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XURCROH_REQUIRE_MLOCK", "1")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC(1))
    with pytest.raises(ShardIntegrityError, match="cannot lock"):
        lock_buffer(bytearray(16))
