from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from xurcroh.errors import ConfigurationError
from xurcroh.fortress.checkpoint import artifact_set_digest
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.model_compiler import FortressCheckpoint
from xurcroh.fortress.placement import (
    Operation,
    PlacementMap,
    PlacementPolicy,
    Provider,
    RoleAssignment,
    place_operations,
)
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker_cli import artifact_file_name, load_role_artifacts


def place_rung_operations(
    rung: LadderRung, operations: list[Operation]
) -> tuple[LadderRung, PlacementMap]:
    if not operations:
        raise ConfigurationError("placement requires discovered model operations")
    providers = [
        Provider(worker.name, worker.country, worker.cpu_vendor) for worker in rung.workers
    ]
    policy = PlacementPolicy(
        shares=rung.share_count,
        max_provider_fraction=rung.max_provider_fraction,
    )
    if any(worker.operation_roles for worker in rung.workers):
        assignments = tuple(
            RoleAssignment(name, role[0], role[1], worker.name, 0)
            for worker in rung.workers
            for name, role in worker.operation_roles.items()
        )
        placement = PlacementMap(
            assignments,
            tuple(operations),
            {provider.provider_id: provider for provider in providers},
            policy,
            rung.placement_epoch,
        )
    else:
        placement = place_operations(operations, providers, policy, epoch=rung.placement_epoch)
    discovered = {operation.identifier for operation in operations}
    assigned = {assignment.unit_id for assignment in placement.assignments}
    if assigned != discovered:
        raise ConfigurationError("placement operations do not match the discovered model graph")
    roles: dict[str, dict[str, tuple[int, int]]] = {worker.name: {} for worker in rung.workers}
    for assignment in placement.assignments:
        if assignment.unit_id in roles[assignment.provider_id]:
            raise ConfigurationError("a worker cannot hold duplicate roles for one operation")
        roles[assignment.provider_id][assignment.unit_id] = (
            assignment.activation_share,
            assignment.weight_share,
        )
    workers = tuple(replace(worker, operation_roles=roles[worker.name]) for worker in rung.workers)
    placed = replace(rung, workers=workers, placement_epoch=placement.epoch)
    placed._validate()
    return placed, placement


def write_operation_role_artifacts(
    name: str,
    operation: CompiledOperation,
    rung: LadderRung,
    output: str | Path,
) -> None:
    root = Path(output)
    placed = any(worker.operation_roles for worker in rung.workers)
    for worker in rung.workers:
        assignment = worker.operation_roles.get(name)
        if placed and assignment is None:
            continue
        directory = root / worker.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact_file_name(name)}.json"
        weight_index = assignment[1] if assignment is not None else worker.weight_index
        encoded = encode_role_artifact(
            operation, weight_index, bytes.fromhex(worker.recipient_public)
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)


def write_role_packages(
    compiled: FortressCheckpoint | Mapping[str, CompiledOperation],
    rung: LadderRung,
    output: str | Path,
) -> dict[str, Path]:
    if not any(worker.operation_roles for worker in rung.workers) and len(rung.workers) != (
        rung.share_count**2
    ):
        raise ConfigurationError("an unplaced worker pool cannot receive role artifacts")
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
    role_packages: str | Path | None = None,
) -> dict[str, Any]:
    """Build the client configuration for a rung.

    ``role_packages`` is the directory ``write_role_packages`` filled. When given,
    the client pins the digest of each worker's artifacts as written there.
    """

    placed = any(worker.operation_roles for worker in rung.workers)
    if not placed and len(rung.workers) != rung.share_count**2:
        raise ConfigurationError("an unplaced worker pool cannot configure the SDK")
    cluster = []
    for worker in rung.workers:
        labels = {
            "activation_index": str(worker.activation_index),
            "weight_index": str(worker.weight_index),
            "recipient_public": worker.recipient_public,
            "client_certificate_path": str(certificate_path),
            "client_key_path": str(key_path),
            "ca_path": str(ca_path),
        }
        if worker.identity_public is not None:
            labels["identity_public"] = worker.identity_public
        package = None if role_packages is None else Path(role_packages) / worker.name
        if package is not None and package.is_dir():
            labels["artifact_digest"] = artifact_set_digest(load_role_artifacts(package))
        if placed:
            labels["operations"] = ",".join(sorted(worker.operation_roles))
            labels["operation_roles"] = json.dumps(
                {name: list(role) for name, role in worker.operation_roles.items()},
                sort_keys=True,
                separators=(",", ":"),
            )
        cluster.append(
            {
                "name": worker.name,
                "endpoint": f"quic://{worker.endpoint}",
                "labels": labels,
            }
        )
    definition = {
        "path": str(checkpoint_path),
        "revision": rung.revision,
        "engine": "fortress",
        "modality": rung.modality,
        "allow_download": False,
        "fortress_seed": fortress_seed,
        "fortress_shares": rung.share_count,
    }
    if rung.verification.get("reap_keep_fraction") is not None:
        definition["reap_keep_fraction"] = rung.verification["reap_keep_fraction"]
        definition["reap_saliency_path"] = rung.verification["reap_saliency_path"]
    return {
        "cluster": cluster,
        "model_path": {rung.model: definition},
        "security": {"mode": "strict"},
    }
