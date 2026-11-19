from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from tests.fortress.test_ladder import manifest_mapping

from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_provider import (
    destroy_ladder_instances,
    materialize_ladder_manifest,
)
from xurcroh.providers.vast import StateStore


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
        private = X25519PrivateKey.from_private_bytes(private_path.read_bytes())
        public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        assert public == worker.recipient_public
        assert private_path.stat().st_mode & 0o777 == 0o600


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
        provisioned={"workers": [{"provider_instance_id": 900, "provider_id": "external"}]},
        registry=registry,
    )
    assert destroyed == [900]
    assert result["cost_usd"] == 2.5
    assert result["destruction"] == {
        "all_destroyed": True,
        "remaining_instance_ids": [],
    }
