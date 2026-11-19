#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xurcroh.errors import ProviderError
from xurcroh.providers.registry import ProviderRegistry


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def run_lifecycle(
    *,
    provider: str,
    offer_id: str,
    values_path: Path,
    state_path: Path,
    registry: ProviderRegistry,
    recover: bool = False,
) -> dict[str, Any]:
    if recover:
        state = _load(state_path)
        instance_id = state.get("provider_instance_id")
        offer = state.get("offer")
        if not instance_id or not isinstance(offer, dict):
            raise ProviderError("recovery state has no created instance; manual review is required")
        if state.get("provider_id") != provider or offer.get("id") != offer_id:
            raise ProviderError("recovery state does not match the requested provider and offer")
    else:
        if state_path.exists():
            raise ProviderError("state already exists; use --recover instead of creating again")
        values = _load(values_path)
        offers = [
            offer
            for offer in registry.offers()
            if offer.get("provider_id") == provider and str(offer.get("id")) == offer_id
        ]
        if len(offers) != 1:
            raise ProviderError("the requested provider offer is unavailable")
        offer = offers[0]
        _write_json(
            state_path,
            {"schema_version": 1, "provider_id": provider, "offer": offer, "status": "creating"},
        )
        instance_id = registry.create(offer, **values)
        _write_json(
            state_path,
            {
                "schema_version": 1,
                "provider_id": provider,
                "provider_instance_id": instance_id,
                "offer": offer,
                "status": "created",
            },
        )
    try:
        if not registry.remaining(offer, instance_id):
            raise ProviderError("created instance is not visible before teardown")
        charges_before = registry.charges(offer, instance_id)
    except Exception:
        try:
            registry.destroy(offer, instance_id)
        finally:
            _write_json(
                state_path,
                {
                    "schema_version": 1,
                    "provider_id": provider,
                    "provider_instance_id": instance_id,
                    "offer": offer,
                    "status": "teardown_required",
                },
            )
        raise
    registry.destroy(offer, instance_id)
    if registry.remaining(offer, instance_id):
        _write_json(
            state_path,
            {
                "schema_version": 1,
                "provider_id": provider,
                "provider_instance_id": instance_id,
                "offer": offer,
                "status": "teardown_required",
            },
        )
        raise ProviderError("provider still reports the instance after destruction")
    charges_after = registry.charges(offer, instance_id)
    report = {
        "schema_version": 1,
        "provider_id": provider,
        "provider_instance_id": instance_id,
        "offer_id": offer_id,
        "provider_api_validated": True,
        "physical_attestation_validated": False,
        "remaining": False,
        "charges_before_usd": charges_before,
        "charges_after_usd": charges_after,
        "status": "destroyed",
    }
    _write_json(state_path, {**report, "offer": offer})
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate one cloud provider lifecycle with recoverable mandatory teardown"
    )
    result.add_argument("--provider", required=True)
    result.add_argument("--offer-id", required=True)
    result.add_argument("--values", type=Path, help="non-secret create parameters as JSON")
    result.add_argument("--state", type=Path, required=True)
    result.add_argument("--output", type=Path)
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--recover", action="store_true")
    action.add_argument("--list-offers", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = ProviderRegistry.with_defaults()
    if args.list_offers:
        offers = [
            offer
            for offer in registry.offers()
            if offer.get("provider_id") == args.provider and str(offer.get("id")) == args.offer_id
        ]
        print(json.dumps(offers, indent=2, sort_keys=True))
        return 0 if offers else 1
    if args.create and args.values is None:
        raise ProviderError("--values is required with --create")
    report = run_lifecycle(
        provider=args.provider,
        offer_id=args.offer_id,
        values_path=args.values,
        state_path=args.state,
        registry=registry,
        recover=args.recover,
    )
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
