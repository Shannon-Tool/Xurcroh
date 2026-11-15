from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


class DecoyScheduler:
    def plan(self, *, real_calls: int, total_calls: int) -> list[bool]:
        if real_calls < 0 or total_calls < 0 or real_calls > total_calls:
            raise ValueError("decoy scheduling requires 0 <= real calls <= total calls")
        plan = [True] * real_calls + [False] * (total_calls - real_calls)
        secrets.SystemRandom().shuffle(plan)
        return plan


@dataclass(frozen=True)
class CanaryTrip:
    canary_id: str
    observers: tuple[str, ...]
    tripped_at: str


class CanaryRevocationError(RuntimeError):
    pass


@dataclass
class _CanaryRecord:
    secret: bytearray = field(repr=False)
    observers: tuple[str, ...]
    revoked: set[str] = field(default_factory=set)
    revoking: set[str] = field(default_factory=set)


class CanaryRegistry:
    def __init__(self, *, revoke: Callable[[str], None]) -> None:
        self._revoke = revoke
        self._records: dict[str, _CanaryRecord] = {}
        self._tripped: set[str] = set()
        self._trips: dict[str, CanaryTrip] = {}
        self._lock = threading.Lock()

    @property
    def trips(self) -> dict[str, CanaryTrip]:
        with self._lock:
            return dict(self._trips)

    def register(self, secret: bytes, *, observers: tuple[str, ...]) -> str:
        if len(secret) < 16:
            raise ValueError("canary secrets must contain at least 16 bytes")
        if not observers or len(set(observers)) != len(observers):
            raise ValueError("canary observers must be unique and nonempty")
        canary_id = secrets.token_urlsafe(32)
        with self._lock:
            self._records[canary_id] = _CanaryRecord(bytearray(secret), observers)
        return canary_id

    def observe(self, canary_id: str, observed_secret: bytes) -> bool:
        with self._lock:
            record = self._records.get(canary_id)
            if record is None:
                return False
            expected = hashlib.sha256(record.secret).digest()
            actual = hashlib.sha256(observed_secret).digest()
            if not hmac.compare_digest(expected, actual):
                return False
            observers = tuple(
                observer
                for observer in record.observers
                if observer not in record.revoked and observer not in record.revoking
            )
            if canary_id in self._tripped and not observers:
                return False
            if canary_id not in self._tripped:
                self._tripped.add(canary_id)
                self._trips[canary_id] = CanaryTrip(
                    canary_id, record.observers, datetime.now(UTC).isoformat()
                )
            record.revoking.update(observers)
        failures: list[str] = []
        for observer in observers:
            try:
                self._revoke(observer)
            except Exception as error:
                failures.append(f"{observer} ({type(error).__name__})")
            else:
                with self._lock:
                    record.revoked.add(observer)
            finally:
                with self._lock:
                    record.revoking.discard(observer)
        if failures:
            raise CanaryRevocationError("failed to revoke canary observers: " + ", ".join(failures))
        return True

    def close(self) -> None:
        with self._lock:
            for record in self._records.values():
                for index in range(len(record.secret)):
                    record.secret[index] = 0
            self._records.clear()
