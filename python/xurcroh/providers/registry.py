from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xurcroh.errors import ProviderError


@dataclass(frozen=True)
class ProviderAdapter:
    offers: Callable[[str | None], list[dict[str, Any]]]
    create: Callable[..., int]
    destroy: Callable[[int], None]


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, provider_id: str, adapter: ProviderAdapter) -> None:
        if not provider_id or provider_id in self._adapters:
            raise ProviderError("provider identities must be unique and nonempty")
        self._adapters[provider_id] = adapter

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if not self._adapters:
            raise ProviderError("no provisioning providers are registered")
        offers = []
        for provider_id, adapter in self._adapters.items():
            for offer in adapter.offers(gpu_name):
                offers.append({**offer, "provider_id": provider_id})
        return offers

    def create(self, offer: dict[str, Any], **values: Any) -> int:
        provider_id = str(offer.get("provider_id") or "vast")
        try:
            adapter = self._adapters[provider_id]
        except KeyError as error:
            raise ProviderError(f"unknown provisioning provider: {provider_id}") from error
        return adapter.create(offer, **values)

    def destroy(self, offer: dict[str, Any], instance_id: int) -> None:
        provider_id = str(offer.get("provider_id") or "vast")
        try:
            adapter = self._adapters[provider_id]
        except KeyError as error:
            raise ProviderError(f"unknown provisioning provider: {provider_id}") from error
        adapter.destroy(instance_id)
