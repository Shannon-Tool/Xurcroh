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
    first = configuration["cluster"][0]
    assert first["endpoint"] == f"quic://{rung.workers[0].endpoint}"
    assert first["labels"]["activation_index"] == "0"
    assert first["labels"]["client_key_path"] == str(tmp_path / "client.key")
