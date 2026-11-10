from __future__ import annotations

import hashlib

import pytest

from xurcroh.attestation import AttestationEvidence, AttestationVerifier
from xurcroh.errors import AttestationError


def evidence() -> AttestationEvidence:
    workload = b"workload"
    artifact = b"artifact"
    public = b"public"
    report_data = hashlib.sha256(b"fresh" + workload + artifact + public).hexdigest()
    return AttestationEvidence(
        vendor="amd",
        nonce=b"fresh",
        report_data=report_data,
        workload_hash=workload,
        artifact_hash=artifact,
        public_key_hash=public,
    )


def test_attestation_binding_and_freshness() -> None:
    verifier = AttestationVerifier(expected_nonce=b"fresh")
    assert verifier.verify_binding(evidence()).vendor == "amd"
    with pytest.raises(AttestationError, match="nonce"):
        verifier.verify_binding(evidence().replace(nonce=b"stale"))
    with pytest.raises(AttestationError, match="workload"):
        verifier.verify_binding(evidence().replace(workload_hash=b"tampered"))
