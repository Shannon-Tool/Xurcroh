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
