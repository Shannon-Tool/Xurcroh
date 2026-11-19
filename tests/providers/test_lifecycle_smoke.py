from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from xurcroh.errors import ProviderError
from xurcroh.providers.registry import ProviderAdapter, ProviderRegistry


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/fortress/tee_provider_smoke.py"
    spec = importlib.util.spec_from_file_location("tee_provider_smoke", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def registry(remaining_after_destroy=False):
    calls = []
    remaining = [True]
    registry = ProviderRegistry()
    registry.register(
        "azure",
        ProviderAdapter(
            offers=lambda gpu_name: [{"id": "shape", "provider_id": "azure"}],
            create=lambda offer, **values: calls.append(("create", values)) or "vm-1",
            destroy=lambda instance_id: (
                calls.append(("destroy", instance_id))
                or remaining.__setitem__(0, remaining_after_destroy)
            ),
            charges=lambda offer, instance_id: 1.25,
            remaining=lambda offer, instance_id: remaining[0],
        ),
    )
    return registry, calls


def test_lifecycle_smoke_creates_charges_and_destroys(tmp_path: Path) -> None:
    smoke = module()
    adapters, calls = registry()
    values = tmp_path / "values.json"
    values.write_text(json.dumps({"label": "tee-smoke"}))
    state = tmp_path / "state.json"
    report = smoke.run_lifecycle(
        provider="azure",
        offer_id="shape",
        values_path=values,
        state_path=state,
        registry=adapters,
    )
    assert report["provider_api_validated"] is True
    assert report["physical_attestation_validated"] is False
    assert report["remaining"] is False
    assert report["charges_after_usd"] == 1.25
    assert calls == [("create", {"label": "tee-smoke"}), ("destroy", "vm-1")]
    assert state.stat().st_mode & 0o777 == 0o600


def test_lifecycle_recovery_does_not_create_another_instance(tmp_path: Path) -> None:
    smoke = module()
    adapters, calls = registry()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "provider_id": "azure",
                "provider_instance_id": "vm-1",
                "offer": {"id": "shape", "provider_id": "azure"},
            }
        )
    )
    report = smoke.run_lifecycle(
        provider="azure",
        offer_id="shape",
        values_path=tmp_path / "unused.json",
        state_path=state,
        registry=adapters,
        recover=True,
    )
    assert report["status"] == "destroyed"
    assert calls == [("destroy", "vm-1")]


def test_lifecycle_rejects_duplicate_create(tmp_path: Path) -> None:
    smoke = module()
    adapters, calls = registry()
    state = tmp_path / "state.json"
    state.write_text("{}")
    with pytest.raises(ProviderError, match="--recover"):
        smoke.run_lifecycle(
            provider="azure",
            offer_id="shape",
            values_path=tmp_path / "values.json",
            state_path=state,
            registry=adapters,
        )
    assert calls == []


def test_lifecycle_fails_when_instance_remains(tmp_path: Path) -> None:
    smoke = module()
    adapters, calls = registry(remaining_after_destroy=True)
    values = tmp_path / "values.json"
    values.write_text("{}")
    with pytest.raises(ProviderError, match="still reports"):
        smoke.run_lifecycle(
            provider="azure",
            offer_id="shape",
            values_path=values,
            state_path=tmp_path / "state.json",
            registry=adapters,
        )
    assert calls[-1] == ("destroy", "vm-1")
