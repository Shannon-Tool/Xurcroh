from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from shards.identities import StageIdentities
from xurcroh.errors import ShardIntegrityError
from xurcroh.stage_worker import DenseStage


def memory_contains(pid: int, needle: bytes) -> bool:
    with (
        open(f"/proc/{pid}/maps", encoding="ascii") as maps,
        open(f"/proc/{pid}/mem", "rb", buffering=0) as memory,
    ):
        for line in maps:
            fields = line.split()
            if "r" not in fields[1]:
                continue
            start, end = (int(value, 16) for value in fields[0].split("-"))
            try:
                memory.seek(start)
                data = memory.read(end - start)
            except OSError:
                continue
            if needle in data:
                return True
    return False


def test_worker_credentials_cannot_open_boundary_or_peer_artifacts(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities = StageIdentities.create(2)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    identity_path = tmp_path / "stage-0.identity"
    identity_path.write_bytes(
        identities.private_keys[0].private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    )
    environment = dict(os.environ)
    environment.pop("XURCROH_MASTER_KEY", None)
    process = subprocess.Popen(  # noqa: S603 - fixed local adversarial-test executable
        [
            sys.executable,
            "tests/security/worker_compromise_child.py",
            str(compiled.manifest_path),
            str(identity_path),
        ],
        cwd=Path(__file__).parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 60)
        assert ready, process.stderr.read() if process.stderr is not None else "worker timeout"
        result = json.loads(process.stdout.readline())
        assert result["master_env_present"] is False
        assert result["boundary_opened_with_private_identity"] is False
        assert result["peer_opened_with_own_identity"] is False
        assert result["basis_tensor_names"] == []
        assert memory_contains(process.pid, identities.boundary_key) is False
        peer_private = identities.private_keys[1].private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        assert memory_contains(process.pid, peer_private) is False
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_v1_manifest_and_wrong_recipient_are_rejected(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities = StageIdentities.create(2)
    compiled = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    with pytest.raises(ShardIntegrityError, match="recipient identity"):
        DenseStage(compiled.manifest_path, 1, identities.private_keys[0])

    document = json.loads(compiled.manifest_path.read_text())
    document["version"] = 1
    compiled.manifest_path.write_text(json.dumps(document))
    with pytest.raises(ShardIntegrityError, match="unsupported stage manifest"):
        DenseStage(compiled.manifest_path, 0, identities.private_keys[0])
