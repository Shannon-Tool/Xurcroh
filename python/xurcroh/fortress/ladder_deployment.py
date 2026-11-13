from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xurcroh.errors import ConfigurationError
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.model_compiler import FortressCheckpoint
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker_cli import artifact_file_name


def write_operation_role_artifacts(
    name: str,
    operation: CompiledOperation,
    rung: LadderRung,
    output: str | Path,
) -> None:
    root = Path(output)
    for worker in rung.workers:
        directory = root / worker.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact_file_name(name)}.json"
        encoded = encode_role_artifact(
            operation, worker.weight_index, bytes.fromhex(worker.recipient_public)
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)


def write_role_packages(
    compiled: FortressCheckpoint | Mapping[str, CompiledOperation],
    rung: LadderRung,
    output: str | Path,
) -> dict[str, Path]:
    operations = compiled.operations if isinstance(compiled, FortressCheckpoint) else compiled
    root = Path(output)
    packages: dict[str, Path] = {}
    for worker in rung.workers:
        directory = root / worker.name
        if directory.exists() and any(directory.iterdir()):
            raise ConfigurationError(f"role package directory is not empty: {directory}")
        packages[worker.name] = directory
    for name, operation in operations.items():
        write_operation_role_artifacts(name, operation, rung, root)
    return packages


def sdk_configuration(
    rung: LadderRung,
    *,
    checkpoint_path: str | Path,
    fortress_seed: int,
    certificate_path: str | Path,
    key_path: str | Path,
    ca_path: str | Path,
) -> dict[str, Any]:
    cluster = []
    for worker in rung.workers:
        cluster.append(
            {
                "name": worker.name,
                "endpoint": f"quic://{worker.endpoint}",
                "labels": {
                    "activation_index": str(worker.activation_index),
                    "weight_index": str(worker.weight_index),
                    "recipient_public": worker.recipient_public,
                    "client_certificate_path": str(certificate_path),
                    "client_key_path": str(key_path),
                    "ca_path": str(ca_path),
                },
            }
        )
    return {
        "cluster": cluster,
        "model_path": {
            rung.model: {
                "path": str(checkpoint_path),
                "revision": rung.revision,
                "engine": "fortress",
                "modality": rung.modality,
                "allow_download": False,
                "fortress_seed": fortress_seed,
                "fortress_shares": rung.share_count,
            }
        },
        "security": {"mode": "strict"},
    }
