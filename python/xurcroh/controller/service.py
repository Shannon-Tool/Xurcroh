from __future__ import annotations

import base64
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from xurcroh.controller.identity import Registration
from xurcroh.controller.registry import WorkerRecord, WorkerRegistry
from xurcroh.controller.topology import Circuit, TopologyPlanner
from xurcroh.errors import ClusterUnavailableError


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
        self._circuits: dict[str, tuple[float, Circuit]] = {}
        self._lock = threading.RLock()

    def create(self, request: CircuitRequest) -> Circuit:
        circuit = self.planner.build(
            request.model,
            self.registry.healthy(),
            pipeline_parallel=request.pipeline_parallel,
            tensor_parallel=request.tensor_parallel,
            expert_parallel=request.expert_parallel,
        )
        with self._lock:
            self._circuits[circuit.circuit_id] = (time.time(), circuit)
        return circuit

    def get(self, circuit_id: str) -> Circuit:
        with self._lock:
            try:
                created, circuit = self._circuits[circuit_id]
            except KeyError as error:
                raise ClusterUnavailableError("unknown or expired circuit") from error
            if time.time() - created > self.ttl_seconds:
                self._circuits.pop(circuit_id, None)
                raise ClusterUnavailableError("unknown or expired circuit")
            return circuit

    def remove(self, circuit_id: str) -> None:
        with self._lock:
            self._circuits.pop(circuit_id, None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._circuits)


def create_controller_app(
    registry: WorkerRegistry | None = None,
    *,
    circuit_ttl_seconds: float = 300,
) -> FastAPI:
    worker_registry = registry or WorkerRegistry()
    circuits = CircuitStore(worker_registry, circuit_ttl_seconds)
    relay_client = httpx.Client(timeout=30)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        relay_client.close()
        circuits._circuits.clear()

    app = FastAPI(title="Xurcroh Blind Controller", version="0.1.0", lifespan=lifespan)

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
        record = worker_registry.register(
            request.registration, signature, Ed25519PublicKey.from_public_bytes(public_bytes)
        )
        return {"worker_id": record.worker_id, "status": "registered"}

    @app.get("/v1/workers")
    def workers() -> tuple[WorkerRecord, ...]:
        return worker_registry.healthy()

    @app.get("/v1/workers/{worker_id}/capabilities")
    def capabilities(worker_id: str) -> WorkerRecord:
        return worker_registry.get(worker_id)

    @app.post("/v1/circuits")
    def create_circuit(request: CircuitRequest) -> Circuit:
        return circuits.create(request)

    @app.delete("/v1/circuits/{circuit_id}", status_code=204)
    def delete_circuit(circuit_id: str) -> Response:
        circuits.remove(circuit_id)
        return Response(status_code=204)

    @app.post("/v1/relay/{circuit_id}")
    async def relay(circuit_id: str, request: Request) -> Response:
        circuit = circuits.get(circuit_id)
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
    def clear_stage_caches(circuit_id: str, request_id: str) -> Response:
        circuit = circuits.get(circuit_id)
        for stage in circuit.stages:
            endpoint = stage.worker.endpoint
            if endpoint.startswith("http+dev://"):
                endpoint = "http://" + endpoint.removeprefix("http+dev://")
            relay_client.delete(f"{endpoint.rstrip('/')}/v1/stages/{request_id}").raise_for_status()
        return Response(status_code=204)

    return app
