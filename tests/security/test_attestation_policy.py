from __future__ import annotations

from pathlib import Path

import pytest

from security.test_attestation import evidence
from security.test_remote_attestation import compiled_manifest
from Xurcroh import Xurcroh
from xurcroh.attestation import AttestationPolicy
from xurcroh.errors import AttestationError


def test_default_tee_policy_accepts_one_supported_hardware_root() -> None:
    AttestationPolicy(mode="tee").validate((evidence(),))
    with pytest.raises(AttestationError, match="requires attestation evidence"):
        AttestationPolicy(mode="tee").validate(())
    with pytest.raises(AttestationError, match="unsupported attestation vendor"):
        AttestationPolicy(mode="tee").validate((evidence().replace(vendor="unknown"),))


def test_remote_tee_mode_does_not_silently_skip_attestation(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities, compiled = compiled_manifest(tiny_llama_path, tmp_path)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    client = Xurcroh(
        cluster={"endpoint": "https://127.0.0.1:1"},
        model_path={
            "tiny": {
                "path": str(tiny_llama_path),
                "manifest_path": str(compiled.manifest_path),
                "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
            }
        },
        security={"mode": "tee", "require_attestation": True},
    )
    with pytest.raises(AttestationError, match="attestation evidence"):
        client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
    client.close()
