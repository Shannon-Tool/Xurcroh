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
        **payload,
    }
    destination = directory / f"{rung.lower()}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
