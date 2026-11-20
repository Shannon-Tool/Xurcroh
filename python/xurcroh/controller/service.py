from __future__ import annotations

import base64
import hmac
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from xurcroh.controller.identity import Registration
from xurcroh.controller.registry import WorkerRecord, WorkerRegistry
from xurcroh.controller.topology import Circuit, TopologyPlanner
from xurcroh.errors import ClusterUnavailableError


@dataclass(frozen=True)
class WorkerEnrollment:
    public_key: Ed25519PublicKey
    endpoint: str


class CircuitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    pipeline_parallel: int = Field(default=1, ge=1)
    tensor_parallel: int = Field(default=1, ge=1)
    expert_parallel: int = Field(default=1, ge=1)


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registration: Registration
    signature: str
    public_key: str


class CircuitStore:
    def __init__(self, registry: WorkerRegistry, ttl_seconds: float) -> None:
        self.registry = registry
        self.ttl_seconds = ttl_seconds
        self.planner = TopologyPlanner()
        self._circuits: dict[str, tuple[float, Circuit, str]] = {}
        self._lock = threading.RLock()

    def create(self, request: CircuitRequest, owner: str) -> Circuit:
        circuit = self.planner.build(
            request.model,
            self.registry.healthy(),
            pipeline_parallel=request.pipeline_parallel,
            tensor_parallel=request.tensor_parallel,
            expert_parallel=request.expert_parallel,
        )
        with self._lock:
            self._circuits[circuit.circuit_id] = (time.time(), circuit, owner)
        return circuit

    def get(self, circuit_id: str, owner: str) -> Circuit:
        with self._lock:
            try:
                created, circuit, expected_owner = self._circuits[circuit_id]
            except KeyError as error:
                raise ClusterUnavailableError("unknown or expired circuit") from error
            if time.time() - created > self.ttl_seconds:
                self._circuits.pop(circuit_id, None)
                raise ClusterUnavailableError("unknown or expired circuit")
            if owner != expected_owner:
                raise HTTPException(status_code=403, detail="circuit owner does not match")
            return circuit

    def remove(self, circuit_id: str, owner: str) -> None:
        self.get(circuit_id, owner)
        with self._lock:
            self._circuits.pop(circuit_id, None)

    def _purge_expired(self) -> None:
        now = time.time()
        for circuit_id in [
            circuit_id
            for circuit_id, (created, _, _) in self._circuits.items()
            if now - created > self.ttl_seconds
        ]:
            self._circuits.pop(circuit_id, None)

    def list(self, owner: str) -> tuple[Circuit, ...]:
        with self._lock:
            self._purge_expired()
            return tuple(
                circuit
                for _, circuit, expected_owner in self._circuits.values()
                if owner == expected_owner
            )

    def active_count(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._circuits)


def create_controller_app(
    registry: WorkerRegistry | None = None,
    *,
    circuit_ttl_seconds: float = 300,
    enrollments: Mapping[str, WorkerEnrollment] | None = None,
    client_tokens: Mapping[str, str] | None = None,
    require_authentication: bool = True,
    allow_development_endpoints: bool = False,
) -> FastAPI:
    worker_registry = registry or WorkerRegistry()
    circuits = CircuitStore(worker_registry, circuit_ttl_seconds)
    relay_client = httpx.Client(timeout=30)
    worker_enrollments = dict(enrollments or {})
    authorized_clients = dict(client_tokens or {})

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        relay_client.close()
        circuits._circuits.clear()

    app = FastAPI(title="Xurcroh Blind Controller", version="0.1.0", lifespan=lifespan)

    def owner(request: Request) -> str:
        if not require_authentication:
            return "development"
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme != "Bearer" or not token:
            raise HTTPException(status_code=403, detail="client authentication is required")
        for expected_token, client_owner in authorized_clients.items():
            if hmac.compare_digest(token, expected_token):
                return client_owner
        raise HTTPException(status_code=403, detail="invalid client credentials")

    def validate_enrollment(registration: Registration, public_bytes: bytes) -> None:
        if not require_authentication:
            return
        enrollment = worker_enrollments.get(registration.worker_id)
        if enrollment is None:
            raise HTTPException(status_code=403, detail="worker is not enrolled")
        expected_public = enrollment.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        if not hmac.compare_digest(public_bytes, expected_public):
            raise HTTPException(status_code=403, detail="worker key is not enrolled")
        if registration.endpoint != enrollment.endpoint:
            raise HTTPException(status_code=403, detail="worker endpoint is not enrolled")
        allowed_scheme = registration.endpoint.startswith("https://") or (
            allow_development_endpoints and registration.endpoint.startswith("http+dev://")
        )
        if not allowed_scheme:
            raise HTTPException(status_code=403, detail="worker endpoint scheme is not allowed")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> dict[str, int]:
        return {
            "workers_healthy": len(worker_registry.healthy()),
            "circuits_active": circuits.active_count(),
        }

    @app.post("/v1/workers/register")
    def register(request: RegistrationRequest) -> dict[str, Any]:
        try:
            signature = base64.b64decode(request.signature, validate=True)
            public_bytes = base64.b64decode(request.public_key, validate=True)
        except ValueError as error:
            raise ClusterUnavailableError("invalid registration encoding") from error
        validate_enrollment(request.registration, public_bytes)
        record = worker_registry.register(
            request.registration, signature, Ed25519PublicKey.from_public_bytes(public_bytes)
        )
        return {"worker_id": record.worker_id, "status": "registered"}

    @app.post("/v1/workers/{worker_id}/heartbeat")
    def heartbeat(worker_id: str, request: Request) -> dict[str, Any]:
        owner(request)
        record = worker_registry.heartbeat(worker_id)
        return {"worker_id": record.worker_id, "status": "alive"}

    @app.get("/v1/workers")
    def workers(request: Request) -> tuple[WorkerRecord, ...]:
        owner(request)
        return worker_registry.healthy()

    @app.get("/v1/workers/{worker_id}/capabilities")
    def capabilities(worker_id: str, request: Request) -> WorkerRecord:
        owner(request)
        return worker_registry.get(worker_id)

    @app.get("/v1/circuits")
    def list_circuits(request: Request) -> tuple[Circuit, ...]:
        return circuits.list(owner(request))

    @app.post("/v1/circuits")
    def create_circuit(request: CircuitRequest, http_request: Request) -> Circuit:
        return circuits.create(request, owner(http_request))

    @app.delete("/v1/circuits/{circuit_id}", status_code=204)
    def delete_circuit(circuit_id: str, request: Request) -> Response:
        circuits.remove(circuit_id, owner(request))
        return Response(status_code=204)

    @app.post("/v1/relay/{circuit_id}")
    async def relay(circuit_id: str, request: Request) -> Response:
        circuit = circuits.get(circuit_id, owner(request))
        stage_header = request.headers.get("X-Xurcroh-Stage", "0")
        try:
            stage_index = int(stage_header)
            stage = circuit.stages[stage_index]
        except (ValueError, IndexError) as error:
            raise ClusterUnavailableError("invalid relay stage") from error
        payload = await request.body()
        endpoint = stage.worker.endpoint
        if endpoint.startswith("http+dev://"):
            endpoint = "http://" + endpoint.removeprefix("http+dev://")
        response = relay_client.post(
            f"{endpoint.rstrip('/')}/v1/relay",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        return Response(
            content=response.content,
            media_type="application/octet-stream",
            status_code=response.status_code,
        )

    @app.delete("/v1/relay/{circuit_id}/stages/{request_id}", status_code=204)
    def clear_stage_caches(circuit_id: str, request_id: str, request: Request) -> Response:
        circuit = circuits.get(circuit_id, owner(request))
        for stage in circuit.stages:
            endpoint = stage.worker.endpoint
            if endpoint.startswith("http+dev://"):
                endpoint = "http://" + endpoint.removeprefix("http+dev://")
            relay_client.delete(f"{endpoint.rstrip('/')}/v1/stages/{request_id}").raise_for_status()
        return Response(status_code=204)

    return app
