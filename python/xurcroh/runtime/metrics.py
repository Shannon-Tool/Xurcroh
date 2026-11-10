from __future__ import annotations

import threading


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {
            "requests_started": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "requests_timed_out": 0,
            "requests_cancelled": 0,
            "duration_ms_total": 0.0,
        }

    def started(self) -> None:
        self._increment("requests_started")

    def completed(self, duration_ms: float) -> None:
        self._increment("requests_completed")
        with self._lock:
            self._values["duration_ms_total"] += duration_ms

    def failed(self) -> None:
        self._increment("requests_failed")

    def timed_out(self) -> None:
        self._increment("requests_timed_out")

    def cancelled(self) -> None:
        self._increment("requests_cancelled")

    def _increment(self, name: str) -> None:
        with self._lock:
            self._values[name] += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._values)
