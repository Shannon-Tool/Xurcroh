#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from xurcroh.errors import BudgetExceededError
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
        charges = client.refresh_charges(store)
        try:
            guard.require_below_limit()
        except BudgetExceededError:
            client.destroy_tracked(store, guard)
        snapshot = store.snapshot()
        print(
            json.dumps(
                {
                    "instances": snapshot["instances"],
                    "charges": charges,
                    "total_cost_usd": guard.total(),
                    "target_usd": args.target_usd,
                    "hard_limit_usd": args.hard_limit_usd,
                    "watchdog_pid": snapshot.get("watchdog_pid"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
