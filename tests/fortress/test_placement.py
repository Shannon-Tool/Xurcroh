from __future__ import annotations

import pytest

from xurcroh.fortress.placement import (
    Operation,
    PlacementMap,
    PlacementPolicy,
    Provider,
    place_operations,
)


def providers() -> list[Provider]:
    return [
        Provider(f"provider-{index}", "us" if index % 2 else "de", "amd" if index % 3 else "intel")
        for index in range(40)
    ]


def operations() -> list[Operation]:
    return [
        Operation("query", 0, "projection", paired_with="key"),
        Operation("key", 1, "projection", paired_with="query"),
        Operation("gate", 2, "projection", paired_with="up"),
        Operation("up", 3, "projection", paired_with="gate"),
        Operation("down", 4, "projection"),
    ]


def test_placement_rejects_single_country_and_cpu_vendor_grids() -> None:
    homogeneous = [Provider(f"provider-{index}", "us", "amd") for index in range(4)]
    with pytest.raises(ValueError, match="country and CPU vendor diversity"):
        place_operations(operations()[:1], homogeneous, PlacementPolicy())


def test_placement_backtracks_to_find_country_and_cpu_vendor_diversity() -> None:
    candidates = [
        *(Provider(f"provider-{index}", "us", "amd") for index in range(4)),
        *(Provider(f"provider-{index}", "de", "intel") for index in range(4, 8)),
    ]
    placement = place_operations(operations()[:1], candidates, PlacementPolicy())
    selected = [placement.provider(assignment.provider_id) for assignment in placement.assignments]
    assert len({provider.country for provider in selected}) == 2
    assert len({provider.cpu_vendor for provider in selected}) == 2


def test_placement_enforces_multiple_operation_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xurcroh.fortress.placement.secrets.SystemRandom.shuffle",
        lambda self, values: None,
    )
    paired = [
        Operation("gate", 0, "feed_forward", paired_with=("up", "down")),
        Operation("up", 1, "feed_forward", paired_with=("gate", "down")),
        Operation("down", 2, "feed_forward", paired_with=("gate", "up")),
    ]
    placement = place_operations(
        paired, providers()[:12], PlacementPolicy(max_provider_fraction=1.0)
    )
    by_operation = {
        operation.identifier: {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == operation.identifier
        }
        for operation in paired
    }
    assert by_operation["down"].isdisjoint(by_operation["gate"])
    assert by_operation["down"].isdisjoint(by_operation["up"])


def test_placement_enforces_role_diversity_pairs_replicas_and_epoch_refresh() -> None:
    policy = PlacementPolicy(shares=2, replicas=2, max_provider_fraction=1.0)
    placement = place_operations(operations(), providers(), policy, epoch=3)

    assert placement.epoch == 3
    assert (
        max(
            len(
                {
                    assignment.unit_id
                    for assignment in placement.assignments
                    if assignment.provider_id == provider.provider_id
                }
            )
            for provider in providers()
        )
        <= 5
    )
    for operation in operations():
        roles = [
            assignment
            for assignment in placement.assignments
            if assignment.unit_id == operation.identifier
        ]
        assert len(roles) == 8
        assert len({assignment.provider_id for assignment in roles}) == 8
        assert len({placement.provider(assignment.provider_id).country for assignment in roles}) > 1
        assert (
            len({placement.provider(assignment.provider_id).cpu_vendor for assignment in roles}) > 1
        )
    for left, right in zip(operations(), operations()[1:], strict=False):
        left_providers = {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == left.identifier
        }
        right_providers = {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == right.identifier
        }
        assert left_providers.isdisjoint(right_providers)
    for operation in operations()[:4]:
        paired = next(item for item in operations() if item.identifier == operation.paired_with)
        left = {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == operation.identifier
        }
        right = {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == paired.identifier
        }
        assert left.isdisjoint(right)

    for provider in providers():
        weight_roles = {
            assignment.weight_share
            for assignment in placement.assignments
            if assignment.provider_id == provider.provider_id
        }
        assert len(weight_roles) <= 1

    refreshed = placement.reshuffle()
    assert refreshed.epoch == 4
    assert refreshed.assignments != placement.assignments
    assert isinstance(placement, PlacementMap)
