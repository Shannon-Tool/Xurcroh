from __future__ import annotations

import json
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field


class Registration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    endpoint: str
    models: tuple[str, ...]
    modalities: tuple[str, ...]
    timestamp: float
    engine: str = "vllm"
    group: str | None = None
    gpu_model: str | None = None
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    max_concurrent_requests: int = Field(default=8, ge=1)
    public_metadata: dict[str, Any] = Field(default_factory=dict)

    def canonical(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()


def sign_registration(private_key: Ed25519PrivateKey, registration: Registration) -> bytes:
    return private_key.sign(registration.canonical())


def registration_expired(
    registration: Registration, ttl_seconds: float, *, now: float | None = None
) -> bool:
    current = time.time() if now is None else now
    age = current - registration.timestamp
    return age > ttl_seconds or age < -5
