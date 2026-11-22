from __future__ import annotations

from pathlib import Path

from tests.fortress.test_ladder import manifest_mapping

from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_provider import (
    destroy_ladder_instances,
    materialize_ladder_manifest,
)
from xurcroh.kem import HybridKemKey
from xurcroh.providers.vast import StateStore
from xurcroh.signatures import load_signing_key


def test_provisioned_hosts_generate_valid_role_manifest(tmp_path: Path) -> None:
    template = manifest_mapping()
    del template["workers"]
    template["checkpoint_path"] = "/trusted/model"
    workers = []
    store = StateStore(tmp_path / "state.json")
    for index in range(4):
        workers.append(
            {
                "name": f"worker-{index}",
                "host": f"ssh-{index}.example",
                "ssh_user": "ubuntu",
                "ssh_port": 2200 + index,
                "ssh_key": f"/keys/worker-{index}",
                "public_host": f"192.0.2.{index + 1}",
                "tls_port": 10000 + index,
                "provider_instance_id": index + 1,
                "machine_id": index + 1,
            }
        )
        if index == 3:
            workers[-1].update(
                {
                    "provider_id": "external",
                    "geolocation": "country-external",
                    "cpu_vendor": "amd",
                }
            )
        if index < 3:
            store.add_instance(
                {
                    "id": index + 1,
                    "machine_id": index + 1,
                    "created_at": 1,
                    "compute_hourly_usd": 1,
                    "storage_hourly_usd": 0,
                    "bandwidth_usd": 0,
                    "provider_id": f"provider-{index}",
                    "geolocation": f"country-{index}",
                    "cpu_vendor": "amd" if index % 2 else "intel",
                }
            )
    path = materialize_ladder_manifest(
        template,
        {"workers": workers},
        store,
        tmp_path / "manifest.json",
    )
    document = __import__("json").loads(path.read_text())
    rung = LadderRung.from_mapping(document)
    assert len(rung.workers) == 4
    for index, worker in enumerate(rung.workers):
        assert document["workers"][index]["ssh"] == {
            "host": f"ssh-{index}.example",
            "username": "ubuntu",
            "port": 2200 + index,
            "key_path": f"/keys/worker-{index}",
        }
        private_path = tmp_path / "recipient-keys" / f"{worker.name}.key"
        private = HybridKemKey.from_bytes(private_path.read_bytes())
        public = private.public_key.hex()
        assert public == worker.recipient_public
        assert private_path.stat().st_mode & 0o777 == 0o600
        # Each role also gets a signing identity; only its public half is in the manifest.
        identity_path = tmp_path / "recipient-keys" / f"{worker.name}.identity.key"
        assert load_signing_key(identity_path).public_key.hex() == worker.identity_public
        assert identity_path.stat().st_mode & 0o777 == 0o600


def test_ladder_teardown_reports_actual_cost_and_destruction(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.add_instance(
        {
            "id": 123,
            "machine_id": 1,
            "created_at": 100.0,
            "compute_hourly_usd": 2.0,
            "storage_hourly_usd": 0.0,
            "bandwidth_usd": 0.5,
        }
    )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def refresh_charges(self, store):
            store.record_charges({123: 3.0})

        def destroy_tracked(self, store, guard):
            store.mark_destroyed(123, 3.0)
            return {"all_destroyed": True, "remaining_instance_ids": []}

    import xurcroh.fortress.ladder_provider as module

    original = module.VastClient
    module.VastClient = Client
    try:
        result = destroy_ladder_instances(store, uncapped=True)
    finally:
        module.VastClient = original
    assert result["cost_usd"] == 3.0
    assert result["destruction"]["all_destroyed"] is True


def test_external_provider_teardown_is_verified(tmp_path: Path) -> None:
    from xurcroh.providers.registry import ProviderAdapter, ProviderRegistry

    destroyed = []
    registry = ProviderRegistry()
    registry.register(
        "external",
        ProviderAdapter(
            offers=lambda gpu_name: [],
            create=lambda offer, **values: 900,
            destroy=destroyed.append,
            charges=lambda offer, instance_id: 2.5,
            remaining=lambda offer, instance_id: False,
        ),
    )
    result = destroy_ladder_instances(
        StateStore(tmp_path / "state.json"),
        uncapped=True,
        provisioned={
            "workers": [{"provider_instance_id": "external-vm-900", "provider_id": "external"}]
        },
        registry=registry,
    )
    assert destroyed == ["external-vm-900"]
    assert result["cost_usd"] == 2.5
    assert result["destruction"] == {
        "all_destroyed": True,
        "remaining_instance_ids": [],
    }


def test_builtin_cloud_teardown_uses_default_registry(tmp_path: Path, monkeypatch) -> None:
    from xurcroh.providers.registry import ProviderAdapter, ProviderRegistry

    events = []
    registry = ProviderRegistry()
    registry.register(
        "azure",
        ProviderAdapter(
            offers=lambda gpu_name: [],
            create=lambda offer, **values: "vm-1",
            destroy=lambda instance_id: events.append(("destroy", instance_id)),
            charges=lambda offer, instance_id: events.append(("charges", instance_id)) or 1.25,
            remaining=lambda offer, instance_id: False,
        ),
    )
    monkeypatch.setattr(ProviderRegistry, "with_defaults", classmethod(lambda cls: registry))
    result = destroy_ladder_instances(
        StateStore(tmp_path / "state.json"),
        uncapped=True,
        provisioned={"workers": [{"provider_instance_id": "vm-1", "provider_id": "azure"}]},
    )
    assert events == [("charges", "vm-1"), ("destroy", "vm-1")]
    assert result["cost_usd"] == 1.25
    assert result["destruction"]["all_destroyed"] is True


def test_cloud_teardown_continues_when_charge_query_fails(tmp_path: Path) -> None:
    import pytest

    from xurcroh.errors import ProviderError
    from xurcroh.providers.registry import ProviderAdapter, ProviderRegistry

    destroyed = []
    registry = ProviderRegistry()
    registry.register(
        "ibm",
        ProviderAdapter(
            offers=lambda gpu_name: [],
            create=lambda offer, **values: "vm-1",
            destroy=destroyed.append,
            charges=lambda offer, instance_id: (_ for _ in ()).throw(
                ProviderError("billing unavailable")
            ),
            remaining=lambda offer, instance_id: False,
        ),
    )
    with pytest.raises(ProviderError, match="teardown failed"):
        destroy_ladder_instances(
            StateStore(tmp_path / "state.json"),
            uncapped=True,
            provisioned={"workers": [{"provider_instance_id": "vm-1", "provider_id": "ibm"}]},
            registry=registry,
        )
    assert destroyed == ["vm-1"]
