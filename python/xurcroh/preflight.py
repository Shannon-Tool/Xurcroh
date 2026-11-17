from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import ConfigurationError
from xurcroh.formats import ModelSupportReport, open_checkpoint

MIB = 1024**2


@dataclass(frozen=True)
class CheckpointPreflight(ModelSupportReport):
    source_digest: str
    checkpoint_bytes: int
    checkpoint_layout_digest: str
    estimated_client_peak_bytes: int
    estimated_role_worker_bytes: int
    estimated_grid_weight_bytes: int
    revision_pinned: bool


def checkpoint_storage_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    files = sorted([*path.rglob("*.safetensors"), *path.rglob("*.gguf")])
    if not files:
        raise ConfigurationError(f"checkpoint contains no supported weights: {path}")
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
    if checkpoint.is_dir() or checkpoint.is_file():
        return checkpoint
    if not definition.allow_download:
        raise ConfigurationError(
            f"model {name!r} checkpoint is unavailable: {checkpoint}; "
            "provide a local checkpoint or set allow_download=true"
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


def detect_modality(definition: ModelDefinition) -> str:
    from xurcroh.errors import ModelLoadError
    from xurcroh.formats import detect_checkpoint_format

    checkpoint = Path(definition.path).expanduser()
    try:
        source_format = (
            "safetensors"
            if (checkpoint / "model_index.json").is_file()
            else detect_checkpoint_format(checkpoint)
            if definition.source_format == "auto"
            else definition.source_format
        )
    except ModelLoadError:
        if not definition.allow_download or checkpoint.exists():
            raise
        source_format = "safetensors"
    definition.source_format = source_format
    if source_format != "safetensors":
        if definition.engine not in {"auto", "fortress"}:
            raise ConfigurationError(
                f"{source_format.upper()} checkpoints require the Fortress engine; "
                "plaintext worker engines are not a compatible fallback"
            )
        definition.engine = "fortress"
        if definition.modality == "auto":
            definition.modality = "text"
    if definition.modality != "auto":
        return definition.modality
    index = checkpoint / "model_index.json"
    if index.is_file():
        try:
            document = json.loads(index.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"cannot inspect Diffusers model index: {index}") from error
        description = json.dumps(document).lower()
        definition.modality = (
            "video"
            if any(marker in description for marker in ("video", "wan", "minimax"))
            else "image"
        )
    return definition.modality


def _adapter_available(architecture: str) -> bool:
    if not architecture or architecture == "unknown":
        return False
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    return architecture in CONFIG_MAPPING


def preflight_checkpoint(
    name: str,
    definition: ModelDefinition,
    workers: list[WorkerConfig],
    *,
    require_known_memory: bool = False,
    validate_roles: bool = True,
) -> CheckpointPreflight:
    checkpoint = ensure_checkpoint(name, definition)
    detect_modality(definition)
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
        source_format = "safetensors"
        source_digest = layout_digest
        architecture = str(config.get("_class_name", "unknown"))
        encodings: tuple[str, ...] = ()
    else:
        with open_checkpoint(checkpoint, definition.source_format) as reader:
            config = reader.config
            layout_digest = reader.fingerprint
            source_digest = layout_digest
            source_format = reader.source_format
            architecture = reader.architecture
            encodings = reader.encodings
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
        if definition.engine == "fortress" and not _adapter_available(architecture):
            issues.append(
                f"architecture {architecture!r} has no Fortress semantic adapter; "
                "unknown model graphs are rejected rather than guessed"
            )
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
        if available_gb is None and worker.labels.get("device") == "cpu":
            resource_limit = worker.resources.max_host_memory_gb
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
    if remote and validate_roles and definition.engine == "fortress":
        expected = {
            (activation, weight)
            for activation in range(definition.fortress_shares)
            for weight in range(definition.fortress_shares)
        }
        groups: dict[str, list[tuple[int, int]]] = {}
        recipients: list[str] = []
        for worker in workers:
            try:
                recipient = worker.labels["recipient_public"]
                if len(bytes.fromhex(recipient)) != 32:
                    raise ValueError("recipient key must contain 32 bytes")
                assigned = {
                    name.strip() for name in worker.labels.get("operations", "*").split(",")
                }
                if "operation_roles" in worker.labels:
                    operation_roles = json.loads(worker.labels["operation_roles"])
                    if not isinstance(operation_roles, dict):
                        raise ValueError("operation_roles must be an object")
                    for operation, value in operation_roles.items():
                        if not isinstance(value, list) or len(value) != 2:
                            raise ValueError("operation role must contain two indexes")
                        groups.setdefault(operation, []).append((int(value[0]), int(value[1])))
                else:
                    role = (
                        int(worker.labels["activation_index"]),
                        int(worker.labels["weight_index"]),
                    )
                    for operation in assigned:
                        groups.setdefault(operation, []).append(role)
                recipients.append(recipient)
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"worker {worker.name} has invalid Fortress role metadata: {error}")
            if worker.endpoint is None or not worker.endpoint.startswith("quic://"):
                issues.append(f"worker {worker.name} requires a quic:// Fortress endpoint")
            for label in ("client_certificate_path", "client_key_path", "ca_path"):
                value = worker.labels.get(label)
                if value is None or not Path(value).expanduser().is_file():
                    issues.append(f"worker {worker.name} lacks a readable {label}")
        if not groups:
            groups["*"] = []
        for operation, actual in groups.items():
            if len(actual) != len(expected) or set(actual) != expected:
                issues.append(
                    f"Fortress operation {operation!r} roles must form one complete "
                    f"{definition.fortress_shares}-by-{definition.fortress_shares} grid "
                    "without duplicates"
                )
        if len(recipients) != len(set(recipients)):
            issues.append("Fortress recipient public keys must be unique per role")
    return CheckpointPreflight(
        model=name,
        checkpoint=str(checkpoint),
        source_format=source_format,
        architecture=architecture,
        encodings=encodings,
        supported=not issues,
        issues=tuple(issues),
        required_adapter=architecture,
        source_digest=source_digest,
        checkpoint_bytes=storage_bytes,
        checkpoint_layout_digest=layout_digest,
        estimated_client_peak_bytes=client_peak,
        estimated_role_worker_bytes=role_worker_bytes,
        estimated_grid_weight_bytes=role_worker_bytes * definition.fortress_shares,
        revision_pinned=revision_pinned,
    )
