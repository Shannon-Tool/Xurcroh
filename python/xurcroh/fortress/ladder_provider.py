from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.fortress.ladder import LadderRung
from xurcroh.providers.vast import BudgetGuard, BudgetPolicy, StateStore, VastClient


def materialize_ladder_manifest(
    template: dict[str, Any],
    provisioned: dict[str, Any],
    store: StateStore,
    output: str | Path,
) -> Path:
    workers = provisioned["workers"]
    if not isinstance(workers, list):
        raise ValueError("provisioned workers must be a list")
    records = {int(record["id"]): record for record in store.snapshot()["instances"]}
    destination = Path(output)
    key_directory = destination.parent / "recipient-keys"
    key_directory.mkdir(parents=True, exist_ok=True)
    placements = []
    share_count = int(template["share_count"])
    for index, instance in enumerate(workers):
        record = records[int(instance["provider_instance_id"])]
        private = X25519PrivateKey.generate()
        public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        name = f"role-{index // share_count}-{index % share_count}"
        key_path = key_directory / f"{name}.key"
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(private.private_bytes_raw())
        placements.append(
            {
                "name": name,
                "endpoint": f"{instance['public_host']}:{int(instance['tls_port'])}",
                "ssh": {
                    "host": str(instance["host"]),
                    "username": str(instance["ssh_user"]),
                    "port": int(instance["ssh_port"]),
                    "key_path": str(instance["ssh_key"]),
                },
                "machine_id": int(record["machine_id"]),
                "provider_id": str(record.get("provider_id", "vast")),
                "country": str(record.get("geolocation", "unknown")),
                "cpu_vendor": str(record.get("cpu_vendor", "unknown")),
                "activation_index": index // share_count,
                "weight_index": index % share_count,
                "recipient_public": public.hex(),
            }
        )
    document = {
        **template,
        "workers": placements,
        "recipient_key_directory": str(key_directory),
    }
    rung = LadderRung.from_mapping(document, allow_unverified_hardware=True)
    if len(rung.workers) != len(workers):
        raise ValueError("provisioned worker count does not match the share grid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def destroy_ladder_instances(store: StateStore, *, uncapped: bool) -> dict[str, Any]:
    policy = BudgetPolicy.uncapped() if uncapped else BudgetPolicy()
    guard = BudgetGuard(store, policy)
    with VastClient() as client:
        client.refresh_charges(store)
        destruction = client.destroy_tracked(store, guard)
    return {"destruction": destruction, "cost_usd": guard.total()}
