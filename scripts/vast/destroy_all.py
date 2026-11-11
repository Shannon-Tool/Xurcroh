#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from xurcroh.providers.vast import BudgetGuard, BudgetPolicy, StateStore, VastClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path, nargs="?", default=Path("artifacts/vast/state.json"))
    parser.add_argument("--target-usd", type=float, default=25)
    parser.add_argument("--hard-limit-usd", type=float, default=30)
    args = parser.parse_args()
    store = StateStore(args.state)
    policy = BudgetPolicy(target_usd=args.target_usd, hard_limit_usd=args.hard_limit_usd)
    guard = BudgetGuard(store, policy)
    with VastClient() as client:
        try:
            client.refresh_charges(store)
        finally:
            report = client.destroy_tracked(store, guard)
    snapshot = store.snapshot()
    pid = snapshot.get("watchdog_pid")
    if isinstance(pid, int):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        store.update("watchdog_stopped", watchdog_pid=None)
    report["total_cost_usd"] = guard.total()
    report["charges_finalized"] = False
    output = args.state.with_name("destruction.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
