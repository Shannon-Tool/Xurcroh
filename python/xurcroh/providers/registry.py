from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from xurcroh.errors import ProviderError


@dataclass(frozen=True)
class ProviderAdapter:
    offers: Callable[[str | None], list[dict[str, Any]]]
    create: Callable[..., int]
    destroy: Callable[[int], None]
    charges: Callable[[dict[str, Any], int], float] | None = None
    remaining: Callable[[dict[str, Any], int], bool] | None = None


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    @classmethod
    def with_entry_points(cls) -> ProviderRegistry:
        registry = cls()
        for entry_point in entry_points(group="xurcroh.providers"):
            adapter = entry_point.load()
            if not isinstance(adapter, ProviderAdapter):
                raise ProviderError(
                    f"provider plugin {entry_point.name!r} did not return a ProviderAdapter"
                )
            registry.register(entry_point.name, adapter)
        return registry

    def register(self, provider_id: str, adapter: ProviderAdapter) -> None:
        if not provider_id or provider_id in self._adapters:
            raise ProviderError("provider identities must be unique and nonempty")
        self._adapters[provider_id] = adapter

    def _adapter(self, offer: dict[str, Any]) -> ProviderAdapter:
        provider_id = str(offer.get("provider_id") or "vast")
        try:
            return self._adapters[provider_id]
        except KeyError as error:
            raise ProviderError(f"unknown provisioning provider: {provider_id}") from error

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if not self._adapters:
            raise ProviderError("no provisioning providers are registered")
        offers = []
        for provider_id, adapter in self._adapters.items():
            for offer in adapter.offers(gpu_name):
                offers.append({**offer, "provider_id": provider_id})
        return offers

    def create(self, offer: dict[str, Any], **values: Any) -> int:
        return self._adapter(offer).create(offer, **values)

    def destroy(self, offer: dict[str, Any], instance_id: int) -> None:
        self._adapter(offer).destroy(instance_id)

    def charges(self, offer: dict[str, Any], instance_id: int) -> float:
        adapter = self._adapter(offer)
        if adapter.charges is None:
            raise ProviderError("provider does not implement charge reporting")
        return adapter.charges(offer, instance_id)

    def remaining(self, offer: dict[str, Any], instance_id: int) -> bool:
        adapter = self._adapter(offer)
        if adapter.remaining is None:
            raise ProviderError("provider does not implement remaining-instance verification")
        return adapter.remaining(offer, instance_id)
