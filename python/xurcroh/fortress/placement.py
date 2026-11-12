from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    provider_id: str
    country: str
    cpu_vendor: str


@dataclass(frozen=True)
class Operation:
    identifier: str
    index: int
    kind: str
    paired_with: str | None = None


@dataclass(frozen=True)
class RoleAssignment:
    unit_id: str
    activation_share: int
    weight_share: int
    provider_id: str
    replica: int


@dataclass(frozen=True)
class PlacementPolicy:
    shares: int = 2
    replicas: int = 1
    max_provider_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.shares < 2 or self.replicas < 1 or not 0 < self.max_provider_fraction <= 1:
            raise ValueError("invalid placement policy")


@dataclass(frozen=True)
class PlacementMap:
    assignments: tuple[RoleAssignment, ...]
    operations: tuple[Operation, ...]
    providers: dict[str, Provider]
    policy: PlacementPolicy
    epoch: int

    def provider(self, provider_id: str) -> Provider:
        return self.providers[provider_id]

    def reshuffle(self) -> PlacementMap:
        return place_operations(
            list(self.operations), list(self.providers.values()), self.policy, epoch=self.epoch + 1
        )


def place_operations(
    operations: list[Operation],
    providers: list[Provider],
    policy: PlacementPolicy,
    *,
    epoch: int = 0,
) -> PlacementMap:
    if len({provider.provider_id for provider in providers}) != len(providers):
        raise ValueError("provider identities must be unique")
    if len({operation.identifier for operation in operations}) != len(operations):
        raise ValueError("operation identities must be unique")
    required_per_unit = policy.shares**2 * policy.replicas
    if providers and required_per_unit > len(providers):
        raise ValueError("not enough providers for unique grid-cell roles")
    maximum_units = max(1, int(len(operations) * policy.max_provider_fraction))
    indexed = sorted(enumerate(operations), key=lambda item: item[1].index)

    assignments: list[RoleAssignment] = []
    per_operation: dict[str, set[str]] = {}
    provider_units: dict[str, set[str]] = {provider.provider_id: set() for provider in providers}
    roles = [
        (activation_share, weight_share, replica)
        for activation_share in range(policy.shares)
        for weight_share in range(policy.shares)
        for replica in range(policy.replicas)
    ]
    search_nodes = 0

    def search(position: int, previous_providers: set[str]) -> list[RoleAssignment] | None:
        nonlocal search_nodes
        if position == len(indexed):
            return list(assignments)
        operation = indexed[position][1]
        paired_providers = per_operation.get(operation.paired_with or "", set())
        current: set[str] = set()
        countries: set[str] = set()
        cpu_vendors: set[str] = set()

        def assign_role(role_index: int) -> list[RoleAssignment] | None:
            nonlocal search_nodes
            if role_index == len(roles):
                per_operation[operation.identifier] = current
                result = search(position + 1, current)
                if result is not None:
                    return result
                per_operation.pop(operation.identifier, None)
                return None
            activation_share, weight_share, replica = roles[role_index]
            candidates = [
                provider
                for provider in providers
                if provider.provider_id not in current
                and provider.provider_id not in paired_providers
                and provider.provider_id not in previous_providers
                and len(provider_units[provider.provider_id]) < maximum_units
            ]
            candidates.sort(key=lambda provider: provider.provider_id)
            rotation = epoch % len(candidates) if candidates else 0
            candidates = candidates[rotation:] + candidates[:rotation]
            for provider in candidates:
                search_nodes += 1
                if search_nodes > 250_000:
                    raise ValueError("placement search exceeded its bounded state space")
                assignments.append(
                    RoleAssignment(
                        operation.identifier,
                        activation_share,
                        weight_share,
                        provider.provider_id,
                        replica,
                    )
                )
                current.add(provider.provider_id)
                countries.add(provider.country)
                cpu_vendors.add(provider.cpu_vendor)
                provider_units[provider.provider_id].add(operation.identifier)
                result = assign_role(role_index + 1)
                if result is not None:
                    return result
                assignments.pop()
                current.remove(provider.provider_id)
                countries.discard(provider.country)
                cpu_vendors.discard(provider.cpu_vendor)
                provider_units[provider.provider_id].remove(operation.identifier)
            return None

        return assign_role(0)

    result = search(0, set())
    if result is None:
        raise ValueError("placement constraints cannot be satisfied")
    return PlacementMap(
        tuple(result),
        tuple(operations),
        {provider.provider_id: provider for provider in providers},
        policy,
        epoch,
    )
