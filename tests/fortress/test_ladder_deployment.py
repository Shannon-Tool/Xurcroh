from __future__ import annotations

from pathlib import Path

from tests.fortress.test_ladder import manifest_mapping

from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import sdk_configuration, write_role_packages
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.transport import decode_role_artifact
from xurcroh.fortress.worker_cli import load_role_artifacts


def test_role_packages_match_manifest_and_exclude_client_boundaries(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    packages = write_role_packages(compiled, rung, tmp_path / "roles")
    assert set(packages) == {worker.name for worker in rung.workers}
    for worker in rung.workers:
        artifacts = load_role_artifacts(packages[worker.name])
        assert set(artifacts) == set(compiled.operations)
        for encoded in artifacts.values():
            artifact = decode_role_artifact(encoded)
            assert artifact.weight_index == worker.weight_index
            assert artifact.recipient_public.hex() == worker.recipient_public
        assert "model.embed_tokens.weight" not in artifacts
        assert "lm_head.weight" not in artifacts


def test_streaming_role_packaging_discards_worker_matrices(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from xurcroh.fortress.ladder_deployment import write_operation_role_artifacts

    rung = LadderRung.from_mapping(manifest_mapping())
    root = tmp_path / "roles"
    compiled = compile_fortress_checkpoint(
        tiny_llama_path,
        seed=731,
        load_model=False,
        operation_sink=lambda name, operation: write_operation_role_artifacts(
            name, operation, rung, root
        ),
    )
    assert all(
        share.matrix is None
        for operation in compiled.operations.values()
        for share in operation.shares.values()
    )
    assert all(not operation.storage for operation in compiled.operations.values())
    for worker in rung.workers:
        artifacts = load_role_artifacts(root / worker.name)
        assert set(artifacts) == set(compiled.operations)


def placed_rung(checkpoint: Path):
    from xurcroh.fortress.model_compiler import discover_fortress_operations
    from xurcroh.fortress.placement import PlacementPolicy, Provider, place_operations

    document = manifest_mapping()
    workers = []
    providers = []
    for index in range(24):
        provider = Provider(
            f"provider-{index}", f"country-{index % 2}", "amd" if index % 2 else "intel"
        )
        providers.append(provider)
        workers.append(
            {
                "name": provider.provider_id,
                "endpoint": f"192.0.2.{index + 1}:9444",
                "machine_id": index + 1,
                "provider_id": provider.provider_id,
                "country": provider.country,
                "cpu_vendor": provider.cpu_vendor,
                "activation_index": index % 2,
                "weight_index": (index // 2) % 2,
                "recipient_public": f"{index + 1:02x}" * 32,
            }
        )
    placement = place_operations(
        discover_fortress_operations(checkpoint),
        providers,
        PlacementPolicy(shares=2, max_provider_fraction=0.25),
    )
    for worker in workers:
        worker["operation_roles"] = {}
    for assignment in placement.assignments:
        worker = next(worker for worker in workers if worker["name"] == assignment.provider_id)
        worker["operation_roles"][assignment.unit_id] = [
            assignment.activation_share,
            assignment.weight_share,
        ]
    document["workers"] = workers
    document["placement_epoch"] = placement.epoch
    return LadderRung.from_mapping(document), placement


def test_placed_role_packages_contain_only_assigned_operations(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    rung, placement = placed_rung(tiny_llama_path)
    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    packages = write_role_packages(compiled, rung, tmp_path / "roles")
    for worker in rung.workers:
        artifacts = load_role_artifacts(packages[worker.name])
        assert set(artifacts) == set(worker.operation_roles)
        assert set(artifacts) != set(compiled.operations)
        operations = {operation.identifier: operation for operation in placement.operations}
        for name in artifacts:
            paired = operations[name].paired_with
            if isinstance(paired, str):
                assert paired not in artifacts
            elif paired is not None:
                assert set(paired).isdisjoint(artifacts)
        for name, encoded in artifacts.items():
            artifact = decode_role_artifact(encoded)
            assert artifact.weight_index == worker.operation_roles[name][1]
            assert artifact.recipient_public.hex() == worker.recipient_public
    for operation in placement.operations:
        assigned = {
            assignment.provider_id
            for assignment in placement.assignments
            if assignment.unit_id == operation.identifier
        }
        assert len(assigned) == 4


def test_deployment_rejects_stale_operation_placement(tiny_llama_path: Path) -> None:
    import pytest

    from xurcroh.errors import ConfigurationError
    from xurcroh.fortress.ladder_deployment import place_rung_operations
    from xurcroh.fortress.model_compiler import discover_fortress_operations

    rung, _ = placed_rung(tiny_llama_path)
    rung.workers[0].operation_roles["stale.operation"] = (0, 0)
    with pytest.raises(ConfigurationError, match="discovered model graph"):
        place_rung_operations(rung, discover_fortress_operations(tiny_llama_path))


def test_sdk_roles_match_operation_specific_placement(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from xurcroh.fortress.ladder_verifier import _roles

    rung, placement = placed_rung(tiny_llama_path)
    configuration = sdk_configuration(
        rung,
        checkpoint_path=tmp_path / "model",
        fortress_seed=731,
        certificate_path=tmp_path / "client.crt",
        key_path=tmp_path / "client.key",
        ca_path=tmp_path / "ca.crt",
    )
    operation = placement.operations[0].identifier
    roles = _roles(configuration["cluster"], operation)
    expected = {
        (assignment.activation_share, assignment.weight_share): assignment.provider_id
        for assignment in placement.assignments
        if assignment.unit_id == operation
    }
    assert {
        (role.activation_index, role.weight_index): role.provider_id for role in roles
    } == expected


def test_deployment_generates_and_serializes_placement(tiny_llama_path: Path) -> None:
    from xurcroh.fortress.ladder_deployment import place_rung_operations
    from xurcroh.fortress.model_compiler import discover_fortress_operations

    document = manifest_mapping()
    workers = []
    for index in range(24):
        workers.append(
            {
                "name": f"provider-{index}",
                "endpoint": f"192.0.2.{index + 1}:9444",
                "machine_id": index + 1,
                "provider_id": f"provider-{index}",
                "country": f"country-{index % 2}",
                "cpu_vendor": "amd" if index % 2 else "intel",
                "activation_index": index % 2,
                "weight_index": (index // 2) % 2,
                "recipient_public": f"{index + 1:02x}" * 32,
            }
        )
    document["workers"] = workers
    rung = LadderRung.from_mapping(document)
    placed, placement = place_rung_operations(rung, discover_fortress_operations(tiny_llama_path))
    assert placement.policy.max_provider_fraction == 0.25
    assert placement.epoch == placed.placement_epoch
    assert {name for worker in placed.workers for name in worker.operation_roles} == {
        operation.identifier for operation in placement.operations
    }


def test_ladder_sdk_configuration_uses_real_roles_and_client_identity(tmp_path: Path) -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    configuration = sdk_configuration(
        rung,
        checkpoint_path=tmp_path / "model",
        fortress_seed=731,
        certificate_path=tmp_path / "client.crt",
        key_path=tmp_path / "client.key",
        ca_path=tmp_path / "ca.crt",
    )
    assert len(configuration["cluster"]) == 4
    assert configuration["model_path"]["gemma-e2b"]["engine"] == "fortress"
    assert configuration["model_path"]["gemma-e2b"]["fortress_seed"] == 731
    assert configuration["model_path"]["gemma-e2b"]["fortress_shares"] == 2
    first = configuration["cluster"][0]
    assert first["endpoint"] == f"quic://{rung.workers[0].endpoint}"
    assert first["labels"]["activation_index"] == "0"
    assert first["labels"]["client_key_path"] == str(tmp_path / "client.key")
