from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xurcroh.config import WorkerConfig


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
    paired_with: str | tuple[str, ...] | None = None


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


def apply_placement(workers: list[WorkerConfig], placement: PlacementMap) -> list[WorkerConfig]:
    by_name = {worker.name: worker for worker in workers}
    if len(by_name) != len(workers):
        raise ValueError("worker names must be unique")
    unknown = {assignment.provider_id for assignment in placement.assignments} - set(by_name)
    if unknown:
        raise ValueError(f"placement references unknown workers: {sorted(unknown)}")
    expected = {
        (activation, weight)
        for activation in range(placement.policy.shares)
        for weight in range(placement.policy.shares)
    }
    for operation in placement.operations:
        actual = {
            (assignment.activation_share, assignment.weight_share)
            for assignment in placement.assignments
            if assignment.unit_id == operation.identifier
        }
        if actual != expected:
            raise ValueError(f"operation {operation.identifier!r} has an incomplete placement grid")

    placed: list[WorkerConfig] = []
    for worker in workers:
        roles: dict[str, list[int]] = {}
        for assignment in placement.assignments:
            if assignment.provider_id != worker.name:
                continue
            if assignment.unit_id in roles:
                raise ValueError("a worker cannot hold duplicate roles for one operation")
            roles[assignment.unit_id] = [assignment.activation_share, assignment.weight_share]
        configured = worker.model_copy(deep=True)
        configured.labels["operations"] = ",".join(sorted(roles))
        configured.labels["operation_roles"] = json.dumps(
            roles, sort_keys=True, separators=(",", ":")
        )
        placed.append(configured)
    return placed


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
    if (
        len({provider.country for provider in providers}) < 2
        or len({provider.cpu_vendor for provider in providers}) < 2
    ):
        raise ValueError("placement requires country and CPU vendor diversity")
    required_per_unit = policy.shares**2 * policy.replicas
    if providers and required_per_unit > len(providers):
        raise ValueError("not enough providers for unique grid-cell roles")
    maximum_units = max(1, int(len(operations) * policy.max_provider_fraction))
    if len(providers) * maximum_units < required_per_unit * len(operations):
        raise ValueError("not enough provider capacity for per-operation role assignments")
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
    shuffle = secrets.SystemRandom().shuffle
    shuffled_providers = list(providers)
    shuffle(shuffled_providers)
    provider_weight_roles = {
        provider.provider_id: index % policy.shares
        for index, provider in enumerate(shuffled_providers)
    }

    def search(position: int, previous_providers: set[str]) -> list[RoleAssignment] | None:
        nonlocal search_nodes
        if position == len(indexed):
            return list(assignments)
        operation = indexed[position][1]
        if operation.paired_with is None:
            paired_operations: tuple[str, ...] = ()
        elif isinstance(operation.paired_with, str):
            paired_operations = (operation.paired_with,)
        else:
            paired_operations = operation.paired_with
        paired_providers = {
            provider
            for identifier in paired_operations
            for provider in per_operation.get(identifier, set())
        }
        current: set[str] = set()

        def assign_role(role_index: int) -> list[RoleAssignment] | None:
            nonlocal search_nodes
            if role_index == len(roles):
                selected = [provider for provider in providers if provider.provider_id in current]
                if (
                    len({provider.country for provider in selected}) < 2
                    or len({provider.cpu_vendor for provider in selected}) < 2
                ):
                    return None
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
                and provider_weight_roles.get(provider.provider_id, weight_share) == weight_share
            ]
            shuffle(candidates)
            candidates.sort(key=lambda provider: len(provider_units[provider.provider_id]))
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
                provider_units[provider.provider_id].add(operation.identifier)
                had_weight_role = provider.provider_id in provider_weight_roles
                provider_weight_roles[provider.provider_id] = weight_share
                result = assign_role(role_index + 1)
                if result is not None:
                    return result
                assignments.pop()
                current.remove(provider.provider_id)
                provider_units[provider.provider_id].remove(operation.identifier)
                if not had_weight_role:
                    provider_weight_roles.pop(provider.provider_id)
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
