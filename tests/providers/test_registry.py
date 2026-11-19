from __future__ import annotations

from xurcroh.providers.registry import ProviderAdapter, ProviderRegistry


def test_provider_registry_aggregates_backends_and_routes_lifecycle() -> None:
    created = []
    destroyed = []
    registry = ProviderRegistry()
    registry.register(
        "vast",
        ProviderAdapter(
            offers=lambda gpu_name: [{"id": 1, "machine_id": 10, "dph_total": 0.2}],
            create=lambda offer, **values: created.append((offer["id"], values)) or 101,
            destroy=lambda instance_id: destroyed.append(instance_id),
        ),
    )
    registry.register(
        "existing",
        ProviderAdapter(
            offers=lambda gpu_name: [{"id": 2, "machine_id": 20, "dph_total": 0.0}],
            create=lambda offer, **values: created.append((offer["id"], values)) or 202,
            destroy=lambda instance_id: destroyed.append(instance_id),
        ),
    )
    offers = registry.offers()
    assert {offer["provider_id"] for offer in offers} == {"vast", "existing"}
    assert registry.create(offers[0], image="test") == 101
    registry.destroy(offers[1], 202)
    assert created == [(1, {"image": "test"})]
    assert destroyed == [202]


def test_provider_plugins_report_charges_and_remaining_instances(
    monkeypatch,
) -> None:
    import xurcroh.providers.registry as module

    adapter = ProviderAdapter(
        offers=lambda gpu_name: [],
        create=lambda offer, **values: 900,
        destroy=lambda instance_id: None,
        charges=lambda offer, instance_id: 2.5,
        remaining=lambda offer, instance_id: False,
    )
    entry_point = type("EntryPoint", (), {"name": "external", "load": lambda self: adapter})()
    monkeypatch.setattr(module, "entry_points", lambda group: [entry_point])
    registry = ProviderRegistry.with_defaults()
    offer = {"provider_id": "external"}
    assert registry.charges(offer, 900) == 2.5
    assert registry.remaining(offer, 900) is False


def test_default_registry_includes_confidential_cloud_adapters() -> None:
    registry = ProviderRegistry.with_defaults()
    assert registry._adapter({"provider_id": "azure"}) is not None
    assert registry._adapter({"provider_id": "gcp"}) is not None
    assert registry._adapter({"provider_id": "alibaba"}) is not None
