from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from xurcroh.errors import ConfigurationError, ModelNotFoundError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GPUConfig(StrictModel):
    model: str | None = None
    count: int = Field(default=1, ge=1)
    memory_gb: float | None = Field(default=None, gt=0)
    identifiers: list[int] = Field(default_factory=list)
    compute_capability: str | None = None


class ResourceConfig(StrictModel):
    max_concurrent_requests: int = Field(default=8, ge=1, le=4096)
    max_host_memory_gb: float | None = Field(default=None, gt=0)
    max_gpu_memory_gb: float | None = Field(default=None, gt=0)
    cpu_threads: int | None = Field(default=None, ge=1)


class SecurityConfig(StrictModel):
    mode: Literal["strict", "tee", "development"] = "strict"
    require_attestation: bool = False
    allow_plaintext_worker_weights: bool = False
    allow_unauthenticated_transport: bool = False
    memory_scrubbing: bool = True

    @model_validator(mode="after")
    def validate_security_downgrades(self) -> Self:
        if self.mode != "development" and (
            self.allow_plaintext_worker_weights or self.allow_unauthenticated_transport
        ):
            raise ConfigurationError(
                "plaintext weights and unauthenticated transport require development mode"
            )
        if self.mode == "tee" and not self.require_attestation:
            raise ConfigurationError("TEE mode requires attestation")
        return self


class SSHConfig(StrictModel):
    username: str
    port: int = Field(default=22, ge=1, le=65535)
    key_path: str | None = None
    password: SecretStr | None = None
    password_env: str | None = None
    known_hosts_path: str | None = None
    strict_host_key_checking: bool = True

    @field_validator("key_path", "known_hosts_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | None) -> str | None:
        return None if value is None else str(Path(value).expanduser())


class WorkerConfig(StrictModel):
    name: str = "worker"
    host: str | None = None
    endpoint: str | None = None
    ssh: SSHConfig | None = None
    gpu: GPUConfig | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    group: str | None = None
    provider: dict[str, str | int | float | bool] = Field(default_factory=dict)
    network_interface: str | None = None
    credential_env: str | None = None
    models: list[str] = Field(default_factory=list)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)

    @model_validator(mode="after")
    def validate_address(self) -> Self:
        if self.host is None and self.endpoint is None:
            raise ConfigurationError(f"worker {self.name!r} requires host or endpoint")
        if self.endpoint is not None and "://" not in self.endpoint:
            raise ConfigurationError(f"worker {self.name!r} endpoint must include a scheme")
        return self


class ClusterConfig(StrictModel):
    workers: list[WorkerConfig]
    pipeline_parallel: int = Field(default=1, ge=1)
    tensor_parallel: int = Field(default=1, ge=1)
    expert_parallel: int = Field(default=1, ge=1)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    provision_workers: bool = False
    controller_endpoint: str | None = None
    request_timeout_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if not self.workers:
            raise ConfigurationError("cluster requires at least one worker")
        names = [worker.name for worker in self.workers]
        if len(names) != len(set(names)):
            raise ConfigurationError("worker names must be unique")
        if self.pipeline_parallel > len(self.workers):
            raise ConfigurationError("pipeline_parallel cannot exceed available workers")
        if self.provision_workers and self.controller_endpoint is None:
            raise ConfigurationError("provision_workers requires controller_endpoint")
        return self

    @classmethod
    def normalize(
        cls,
        value: object = None,
        *,
        pipeline_parallel: int = 1,
        tensor_parallel: int = 1,
        expert_parallel: int = 1,
        security: SecurityConfig | Mapping[str, object] | None = None,
        provision_workers: bool = False,
    ) -> ClusterConfig:
        advanced: dict[str, object] = {}
        raw_workers: object
        if value is None or value == {}:
            raw_workers = [{"name": "local", "endpoint": "local://cpu"}]
        elif isinstance(value, cls):
            return value
        elif isinstance(value, str):
            raw_workers = [value]
        elif isinstance(value, Mapping) and "workers" in value:
            advanced = dict(value)
            raw_workers = advanced.pop("workers")
        elif isinstance(value, Mapping):
            worker_values = dict(value)
            advanced = {
                key: worker_values.pop(key)
                for key in (
                    "pipeline_parallel",
                    "tensor_parallel",
                    "expert_parallel",
                    "security",
                    "provision_workers",
                    "controller_endpoint",
                    "request_timeout_seconds",
                )
                if key in worker_values
            }
            raw_workers = [worker_values]
        elif isinstance(value, list | tuple):
            raw_workers = value
        else:
            raise ConfigurationError(
                "cluster must be an endpoint, worker mapping, worker list, or configuration mapping"
            )

        if not isinstance(raw_workers, list | tuple):
            raise ConfigurationError("cluster workers must be a list")
        workers = []
        for index, raw in enumerate(raw_workers):
            if isinstance(raw, WorkerConfig):
                worker = raw
            elif isinstance(raw, str):
                worker = WorkerConfig(name=f"worker-{index}", endpoint=raw)
            elif isinstance(raw, Mapping):
                values = dict(raw)
                values.setdefault("name", f"worker-{index}")
                worker = WorkerConfig.model_validate(values)
            else:
                raise ConfigurationError(
                    "each worker must be an endpoint string or configuration mapping"
                )
            workers.append(worker)

        selected_security = security or advanced.pop("security", None) or SecurityConfig()
        try:
            return cls.model_validate(
                {
                    "workers": workers,
                    "pipeline_parallel": advanced.pop("pipeline_parallel", pipeline_parallel),
                    "tensor_parallel": advanced.pop("tensor_parallel", tensor_parallel),
                    "expert_parallel": advanced.pop("expert_parallel", expert_parallel),
                    "security": selected_security,
                    "provision_workers": advanced.pop("provision_workers", provision_workers),
                    "request_timeout_seconds": advanced.pop("request_timeout_seconds", 60),
                    **advanced,
                }
            )
        except ValidationError as error:
            raise ConfigurationError(str(error)) from error


class ModelDefinition(StrictModel):
    path: str
    revision: str | None = None
    engine: Literal[
        "auto", "reference", "transformers", "vllm", "sglang", "diffusion", "xdit", "fortress"
    ] = "auto"
    modality: Literal["auto", "text", "vlm", "image", "video"] = "auto"
    allow_download: bool = False
    manifest_path: str | None = None
    boundary_key_env: str | None = None
    protect_latents: bool = False
    latent_security: Literal["equivariant", "transport_only"] = "equivariant"
    fortress_seed: int | None = None


class ModelRegistry:
    def __init__(self, models: Mapping[str, ModelDefinition]) -> None:
        self._models = MappingProxyType(dict(models))

    @classmethod
    def normalize(cls, value: str | Path | Mapping[str, object]) -> ModelRegistry:
        raw: Mapping[str, object]
        if isinstance(value, str | Path):
            path = str(value)
            name = Path(path).name or "model"
            raw = {name: path}
        elif isinstance(value, Mapping):
            raw = value
        else:
            raise ConfigurationError("model_path must be a path or a named model registry")
        if not raw:
            raise ConfigurationError("model_path must register at least one model")
        models: dict[str, ModelDefinition] = {}
        for name, definition in raw.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", name):
                raise ConfigurationError(f"invalid model name: {name!r}")
            if isinstance(definition, ModelDefinition):
                models[name] = definition
            elif isinstance(definition, str | Path):
                models[name] = ModelDefinition(path=str(definition))
            elif isinstance(definition, Mapping):
                models[name] = ModelDefinition.model_validate(definition)
            else:
                raise ConfigurationError(f"model {name!r} must be a path or model configuration")
        return cls(models)

    def names(self) -> tuple[str, ...]:
        return tuple(self._models)

    def get(self, name: str) -> ModelDefinition:
        try:
            return self._models[name]
        except KeyError as error:
            available = ", ".join(self.names())
            raise ModelNotFoundError(
                f"unknown model {name!r}; available models: {available}"
            ) from error

    def items(self) -> tuple[tuple[str, ModelDefinition], ...]:
        return tuple(self._models.items())
