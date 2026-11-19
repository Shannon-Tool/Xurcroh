from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Protocol

from xurcroh.errors import ProviderError

ProviderInstanceId = str | int


class ConfidentialCloudClient(Protocol):
    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]: ...

    def create(self, offer: dict[str, Any], **values: Any) -> ProviderInstanceId: ...

    def destroy(self, instance_id: str) -> None: ...

    def remaining(self, instance_id: str) -> bool: ...

    def charges(self, instance_id: str) -> float: ...


@dataclass(frozen=True)
class ProviderAdapter:
    offers: Callable[[str | None], list[dict[str, Any]]]
    create: Callable[..., ProviderInstanceId]
    destroy: Callable[[ProviderInstanceId], None]
    charges: Callable[[dict[str, Any], ProviderInstanceId], float] | None = None
    remaining: Callable[[dict[str, Any], ProviderInstanceId], bool] | None = None


def _cloud_adapter(client_factory: Callable[[], ConfidentialCloudClient]) -> ProviderAdapter:
    def offers(gpu_name: str | None) -> list[dict[str, Any]]:
        return client_factory().offers(gpu_name)

    def create(offer: dict[str, Any], **values: Any) -> ProviderInstanceId:
        return client_factory().create(offer, **values)

    def destroy(instance_id: ProviderInstanceId) -> None:
        client_factory().destroy(str(instance_id))

    def charges(offer: dict[str, Any], instance_id: ProviderInstanceId) -> float:
        del offer
        return client_factory().charges(str(instance_id))

    def remaining(offer: dict[str, Any], instance_id: ProviderInstanceId) -> bool:
        del offer
        return client_factory().remaining(str(instance_id))

    return ProviderAdapter(
        offers=offers,
        create=create,
        destroy=destroy,
        charges=charges,
        remaining=remaining,
    )


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    @classmethod
    def with_defaults(cls) -> ProviderRegistry:
        registry = cls()
        from xurcroh.providers.alibaba import AlibabaConfidentialClient
        from xurcroh.providers.aws import AwsNitroClient
        from xurcroh.providers.azure import AzureConfidentialClient
        from xurcroh.providers.gcp import GoogleConfidentialClient
        from xurcroh.providers.ibm import IbmConfidentialClient
        from xurcroh.providers.oci import OciConfidentialClient

        registry.register("azure", _cloud_adapter(AzureConfidentialClient))
        registry.register("gcp", _cloud_adapter(GoogleConfidentialClient))
        registry.register("alibaba", _cloud_adapter(AlibabaConfidentialClient))
        registry.register("oci", _cloud_adapter(OciConfidentialClient))
        registry.register("aws", _cloud_adapter(AwsNitroClient))
        registry.register("ibm", _cloud_adapter(IbmConfidentialClient))
        return registry.with_entry_points(registry)

    @classmethod
    def with_entry_points(cls, registry: ProviderRegistry | None = None) -> ProviderRegistry:
        registry = registry or cls()
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

    def create(self, offer: dict[str, Any], **values: Any) -> ProviderInstanceId:
        return self._adapter(offer).create(offer, **values)

    def destroy(self, offer: dict[str, Any], instance_id: ProviderInstanceId) -> None:
        self._adapter(offer).destroy(instance_id)

    def charges(self, offer: dict[str, Any], instance_id: ProviderInstanceId) -> float:
        adapter = self._adapter(offer)
        if adapter.charges is None:
            raise ProviderError("provider does not implement charge reporting")
        return adapter.charges(offer, instance_id)

    def remaining(self, offer: dict[str, Any], instance_id: ProviderInstanceId) -> bool:
        adapter = self._adapter(offer)
        if adapter.remaining is None:
            raise ProviderError("provider does not implement remaining-instance verification")
        return adapter.remaining(offer, instance_id)
