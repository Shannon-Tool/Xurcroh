from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_fortress_evidence(rung: str, payload: dict[str, Any]) -> None:
    root = os.environ.get("XURCROH_FORTRESS_EVIDENCE_DIR")
    if root is None:
        return
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "rung": rung,
        "revision": os.environ.get("XURCROH_FORTRESS_REVISION", "unknown"),
        "working_tree_dirty": os.environ.get("XURCROH_FORTRESS_WORKING_TREE_DIRTY", "unknown"),
        "recorded_at": datetime.now(UTC).isoformat(),
        "execution": "local CPU; four independent spawned worker processes; real Rust QUIC",
        "billable_resources": False,
        "cost_usd": 0,
        **payload,
    }
    destination = directory / f"{rung.lower()}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def exercise_adversarial_evidence(
    clients: dict[str, Any], roles: list[Any], operations: dict[str, Any]
) -> dict[str, Any]:
    from dataclasses import asdict

    import torch

    from xurcroh.fortress.arithmetic import split_secret
    from xurcroh.fortress.attacks import evaluate_share_reconstruction
    from xurcroh.fortress.canary import CanaryRegistry, DecoyScheduler

    name = next(iter(clients))
    operation = operations[name]
    features = operation.quantized.values.shape[1]
    real_shares = split_secret(torch.arange(features, dtype=torch.int64), count=2).shares
    decoy_shares = split_secret(
        torch.randint(0, 2**32, (features,), dtype=torch.int64), count=2
    ).shares
    observers = tuple(role.provider_id for role in roles if role.activation_index == 0)
    revoked: list[str] = []
    registry = CanaryRegistry(revoke=revoked.append)
    canary_secret = decoy_shares[0].numpy().tobytes()
    canary_id = registry.register(canary_secret, observers=observers)
    for real in DecoyScheduler().plan(real_calls=1, total_calls=2):
        clients[name].execute(real_shares if real else decoy_shares, request_id=f"{name}:mixed")
    tripped = registry.observe(canary_id, canary_secret)
    assert tripped
    assert revoked == list(observers)
    registry.close()
    attack = evaluate_share_reconstruction(
        operation.quantized.values,
        torch.arange(features, dtype=torch.int64),
        [share.matrix for share in operation.shares.values()],
        real_shares,
    )
    return {
        "decoy_calls": 1,
        "decoy_role_requests": 8,
        "canary_tripped": tripped,
        "canary_attributed_observers": revoked,
        "canary_method": "simulated external observation of an actually transmitted share",
        "canary_limitation": "attributes the observer set; does not prove which participant leaked",
        "attack_results": asdict(attack),
    }
