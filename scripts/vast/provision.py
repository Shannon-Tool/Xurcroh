#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from xurcroh.errors import ProviderError
from xurcroh.providers.vast import (
    BudgetGuard,
    BudgetPolicy,
    StateStore,
    VastClient,
    select_diverse_offers,
    write_deployment_config,
)


def budget_policy(args: argparse.Namespace) -> BudgetPolicy:
    if args.uncapped:
        return BudgetPolicy.uncapped()
    return BudgetPolicy(target_usd=args.target_usd, hard_limit_usd=args.hard_limit_usd)


def watchdog(state: Path, ttl_seconds: float, policy: BudgetPolicy) -> None:
    store = StateStore(state)
    guard = BudgetGuard(store, policy)
    deadline = time.time() + ttl_seconds
    failures = 0
    while True:
        time.sleep(30)
        try:
            with VastClient() as client:
                if not store.snapshot()["instances"]:
                    return
                client.refresh_charges(store)
                guard.require_below_limit()
                if time.time() >= deadline:
                    raise ProviderError("validation TTL expired")
                failures = 0
        except Exception:
            failures += 1
            if failures < 3 and time.time() < deadline:
                continue
            try:
                with VastClient() as client:
                    client.destroy_tracked(store, guard)
                return
            except Exception:
                time.sleep(10)


def provision(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ProviderError("host count must be positive")
    store = StateStore(args.state)
    policy = budget_policy(args)
    guard = BudgetGuard(store, policy)
    if any("destroyed_at" not in item for item in store.snapshot()["instances"]):
        raise ProviderError(
            "active tracked instances already exist; destroy them before provisioning"
        )
    with VastClient() as client:
        client.refresh_charges(store)
        offers = client.offers(args.gpu_name)
        eligible = [
            offer
            for offer in offers
            if float(offer.get("gpu_ram", 0)) >= args.min_gpu_ram_gb * 1024
            and float(offer["dph_total"]) <= args.max_hourly_usd
        ]
        selected, forecasts = select_diverse_offers(
            eligible,
            count=args.count,
            guard=guard,
            planned_hours=args.planned_hours,
            disk_gb=args.disk_gb,
            max_country_fraction=args.max_country_fraction,
            max_cpu_vendor_fraction=args.max_cpu_vendor_fraction,
            max_provider_fraction=args.max_provider_fraction,
        )
        created = []
        try:
            for index, offer in enumerate(selected):
                instance_id = client.create(
                    offer,
                    image=args.image,
                    disk_gb=args.disk_gb,
                    label=f"xurcroh-validation-{index}-{int(time.time())}",
                )
                store.add_instance(
                    {
                        "id": instance_id,
                        "offer_id": offer["id"],
                        "machine_id": offer["machine_id"],
                        "gpu_name": offer.get("gpu_name"),
                        "created_at": time.time(),
                        "compute_hourly_usd": float(offer["dph_total"]),
                        "storage_hourly_usd": float(offer.get("storage_total_cost", 0.0)),
                        "disk_gb": args.disk_gb,
                        "bandwidth_usd": 0.0,
                        "planned_cost_usd": guard.forecast(offer, args.planned_hours, args.disk_gb),
                    }
                )
                created.append(instance_id)
            instances = [
                client.wait_running(instance_id, args.boot_timeout, guard)
                for instance_id in created
            ]
            store.update("instances_ready", instance_ids=created)
        except Exception:
            client.destroy_tracked(store, guard)
            raise
        git = shutil.which("git")
        if git is None:
            raise ProviderError("Git is required to record the deployment revision")
        revision = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        config = write_deployment_config(
            store,
            instances,
            model=args.model,
            checkpoint=args.checkpoint,
            compiled=args.compiled,
            output=args.output / "two-node.json",
            revision=revision,
        )
        (args.output / "instances.json").write_text(
            __import__("json").dumps({"instances": instances, "forecast_usd": forecasts}, indent=2)
            + "\n"
        )
        log_descriptor = os.open(
            args.output / "watchdog.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        with os.fdopen(log_descriptor, "ab") as log:
            process = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(args.state),
                    "--watchdog-child",
                    "--ttl-seconds",
                    str(args.ttl_seconds),
                    "--target-usd",
                    str(args.target_usd),
                    "--hard-limit-usd",
                    str(args.hard_limit_usd),
                    *(["--uncapped"] if args.uncapped else []),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        store.update("watchdog_started", watchdog_pid=process.pid)
        print(__import__("json").dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path, nargs="?", default=Path("artifacts/vast/state.json"))
    parser.add_argument("--watchdog-child", action="store_true")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-country-fraction", type=float, default=0.5)
    parser.add_argument("--max-cpu-vendor-fraction", type=float, default=0.5)
    parser.add_argument("--max-provider-fraction", type=float, default=0.5)
    parser.add_argument("--gpu-name")
    parser.add_argument("--min-gpu-ram-gb", type=float, default=12)
    parser.add_argument("--max-hourly-usd", type=float, default=0.75)
    parser.add_argument("--disk-gb", type=float, default=40)
    parser.add_argument("--planned-hours", type=float, default=1.0)
    parser.add_argument("--boot-timeout", type=float, default=1200)
    parser.add_argument("--ttl-seconds", type=float, default=5400)
    parser.add_argument("--target-usd", type=float, default=25)
    parser.add_argument("--hard-limit-usd", type=float, default=30)
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--image", default="vllm/vllm-openai:v0.15.0")
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/Qwen3-0.6B"))
    parser.add_argument("--compiled", type=Path, default=Path("artifacts/qwen3-compiled"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vast"))
    args = parser.parse_args()
    policy = budget_policy(args)
    if args.watchdog_child:
        watchdog(args.state, args.ttl_seconds, policy)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        provision(args)


if __name__ == "__main__":
    main()
