from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import ConfigurationError
from xurcroh.shards.compiler import CheckpointReader

MIB = 1024**2


@dataclass(frozen=True)
class CheckpointPreflight:
    model: str
    checkpoint: str
    supported: bool
    checkpoint_bytes: int
    checkpoint_layout_digest: str
    estimated_client_peak_bytes: int
    estimated_role_worker_bytes: int
    estimated_grid_weight_bytes: int
    revision_pinned: bool
    issues: tuple[str, ...]


def checkpoint_storage_bytes(path: Path) -> int:
    files = sorted(path.rglob("*.safetensors"))
    if not files:
        raise ConfigurationError(f"checkpoint contains no safetensors weights: {path}")
    return sum(path.stat().st_size for path in files if path.is_file())


def checkpoint_value_bytes(config: dict[str, object]) -> int:
    value = config.get("dtype", config.get("torch_dtype", "float32"))
    normalized = str(value).lower().replace("torch.", "")
    return {
        "float64": 8,
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "int8": 1,
        "uint8": 1,
    }.get(normalized, 4)


def preflight_checkpoint(
    name: str,
    definition: ModelDefinition,
    workers: list[WorkerConfig],
) -> CheckpointPreflight:
    checkpoint = Path(definition.path).expanduser()
    if not checkpoint.is_dir():
        raise ConfigurationError(
            f"model {name!r} checkpoint directory is unavailable: {checkpoint}"
        )
    reader = CheckpointReader(checkpoint)
    config = reader.config
    issues: list[str] = []
    text = {**config, **config.get("text_config", {})}
    for field in ("hidden_size", "num_attention_heads", "num_hidden_layers"):
        try:
            valid = int(text.get(field, 0)) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            issues.append(f"checkpoint is missing a valid {field}")
    if not reader.tensor_names():
        issues.append("checkpoint contains no tensor metadata")
    storage_bytes = checkpoint_storage_bytes(checkpoint)
    value_bytes = checkpoint_value_bytes(config)
    parameter_estimate = storage_bytes / value_bytes
    role_worker_bytes = math.ceil(parameter_estimate * 4 + 256 * MIB)
    remote = any(worker.endpoint != "local://cpu" for worker in workers)
    client_multiplier = 2 if remote else 4
    client_peak = math.ceil(storage_bytes * client_multiplier + 256 * MIB)
    revision_pinned = bool(
        definition.revision and re.fullmatch(r"[0-9a-f]{40}", definition.revision)
    )
    if remote and definition.engine == "fortress" and not revision_pinned:
        issues.append("remote Fortress deployment requires an immutable 40-character revision")
    for worker in workers:
        available_gb = worker.gpu.memory_gb if worker.gpu is not None else None
        resource_limit = worker.resources.max_gpu_memory_gb
        if resource_limit is not None:
            available_gb = (
                resource_limit if available_gb is None else min(available_gb, resource_limit)
            )
        if remote and available_gb is None:
            issues.append(
                f"worker {worker.name} memory is unknown; configure gpu.memory_gb "
                "or enable SSH resource discovery"
            )
        elif available_gb is not None and available_gb * 1024**3 < role_worker_bytes:
            issues.append(
                f"worker {worker.name} has insufficient known memory: "
                f"requires approximately {role_worker_bytes / 1024**3:.2f} GiB per weight role"
            )
    return CheckpointPreflight(
        model=name,
        checkpoint=str(checkpoint),
        supported=not issues,
        checkpoint_bytes=storage_bytes,
        checkpoint_layout_digest=reader.fingerprint,
        estimated_client_peak_bytes=client_peak,
        estimated_role_worker_bytes=role_worker_bytes,
        estimated_grid_weight_bytes=role_worker_bytes * definition.fortress_shares,
        revision_pinned=revision_pinned,
        issues=tuple(issues),
    )
