from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from xurcroh.controller.identity import Registration, registration_expired
from xurcroh.errors import AttestationError, ClusterUnavailableError


@dataclass
class WorkerRecord:
    worker_id: str
    endpoint: str
    models: tuple[str, ...]
    modalities: tuple[str, ...]
    engine: str = "vllm"
    group: str | None = None
    gpu_model: str | None = None
    gpu_memory_gb: float | None = None
    max_concurrent_requests: int = 8
    registered_at: float = 0
    last_heartbeat: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkerRegistry:
    def __init__(self, *, registration_ttl_seconds: float = 60) -> None:
        self.registration_ttl_seconds = registration_ttl_seconds
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()

    def register(
        self,
        registration: Registration,
        signature: bytes,
        public_key: Ed25519PublicKey,
    ) -> WorkerRecord:
        if registration_expired(registration, self.registration_ttl_seconds):
            raise AttestationError("worker registration is not fresh")
        try:
            public_key.verify(signature, registration.canonical())
        except InvalidSignature as error:
            raise AttestationError("worker registration signature is invalid") from error
        with self._lock:
            existing = self._workers.get(registration.worker_id)
            if existing is not None and existing.endpoint != registration.endpoint:
                raise AttestationError("worker identity is already assigned to another endpoint")
            record = WorkerRecord(
                worker_id=registration.worker_id,
                endpoint=registration.endpoint,
                models=registration.models,
                modalities=registration.modalities,
                engine=registration.engine,
                group=registration.group,
                gpu_model=registration.gpu_model,
                gpu_memory_gb=registration.gpu_memory_gb,
                max_concurrent_requests=registration.max_concurrent_requests,
                registered_at=registration.timestamp,
                last_heartbeat=time.time(),
                metadata=registration.public_metadata,
            )
            self._workers[record.worker_id] = record
            return record

    def heartbeat(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            record = self.get(worker_id)
            record.last_heartbeat = time.time()
            return record

    def get(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            try:
                record = self._workers[worker_id]
            except KeyError as error:
                raise ClusterUnavailableError(f"unknown worker: {worker_id}") from error
            if time.time() - record.last_heartbeat > self.registration_ttl_seconds:
                raise ClusterUnavailableError(f"worker {worker_id} is stale")
            return record

    def healthy(self) -> tuple[WorkerRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._workers.values()
                if time.time() - record.last_heartbeat <= self.registration_ttl_seconds
            )

    def remove(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)
