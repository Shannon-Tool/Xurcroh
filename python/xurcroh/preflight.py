from __future__ import annotations

import hashlib
import json
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


def ensure_checkpoint(name: str, definition: ModelDefinition) -> Path:
    checkpoint = Path(definition.path).expanduser()
    if checkpoint.is_dir():
        return checkpoint
    if not definition.allow_download:
        raise ConfigurationError(
            f"model {name!r} checkpoint directory is unavailable: {checkpoint}; "
            "provide a local safetensors checkpoint or set allow_download=true"
        )
    try:
        from huggingface_hub import HfApi, snapshot_download

        revision = definition.revision
        if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            revision = HfApi().model_info(repo_id=definition.path, revision=definition.revision).sha
        if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ConfigurationError("the model hub did not return an immutable revision")
        resolved = Path(
            snapshot_download(repo_id=definition.path, revision=revision, local_files_only=False)
        )
    except Exception as error:
        if isinstance(error, ConfigurationError):
            raise
        raise ConfigurationError(
            f"cannot download model {name!r} from {definition.path!r}: {error}"
        ) from error
    definition.path = str(resolved)
    definition.revision = revision
    return resolved


def preflight_checkpoint(
    name: str,
    definition: ModelDefinition,
    workers: list[WorkerConfig],
    *,
    require_known_memory: bool = False,
) -> CheckpointPreflight:
    checkpoint = ensure_checkpoint(name, definition)
    issues: list[str] = []
    if definition.modality in {"image", "video"}:
        index_path = checkpoint / "model_index.json"
        try:
            index_bytes = index_path.read_bytes()
            config = json.loads(index_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(
                f"model {name!r} has no readable Diffusers model_index.json"
            ) from error
        layout_digest = hashlib.sha256(index_bytes).hexdigest()
    else:
        reader = CheckpointReader(checkpoint)
        config = reader.config
        layout_digest = reader.fingerprint
        text = {**config, **config.get("text_config", {})}
        aliases = {
            "hidden_size": "n_embd",
            "num_attention_heads": "n_head",
            "num_hidden_layers": "n_layer",
        }
        for field, alias in aliases.items():
            try:
                valid = int(text.get(field, text.get(alias, 0))) > 0
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
    for worker in workers:
        available_gb = worker.gpu.memory_gb if worker.gpu is not None else None
        resource_limit = worker.resources.max_gpu_memory_gb
        if resource_limit is not None:
            available_gb = (
                resource_limit if available_gb is None else min(available_gb, resource_limit)
            )
        if require_known_memory and remote and available_gb is None:
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
        checkpoint_layout_digest=layout_digest,
        estimated_client_peak_bytes=client_peak,
        estimated_role_worker_bytes=role_worker_bytes,
        estimated_grid_weight_bytes=role_worker_bytes * definition.fortress_shares,
        revision_pinned=revision_pinned,
        issues=tuple(issues),
    )
