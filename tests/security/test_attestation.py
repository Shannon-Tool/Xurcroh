from __future__ import annotations

import hashlib

import pytest

from xurcroh.attestation import (
    AttestationEvidence,
    AttestationVerifier,
    IntelDCAPVerifier,
    SNPGuestVerifier,
)
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


def signed_report_data() -> bytes:
    return bytes.fromhex(evidence().report_data) + bytes(32)


def snp_report() -> bytes:
    report = bytearray(0x2A0)
    report[0x50:0x90] = signed_report_data()
    return bytes(report)


def test_sideband_report_data_is_not_hardware_proof() -> None:
    verifier = AttestationVerifier(expected_nonce=b"fresh")
    with pytest.raises(AttestationError, match="hardware-signed"):
        verifier.verify_binding(evidence())


def test_snp_report_data_drives_binding_and_freshness() -> None:
    verifier = SNPGuestVerifier(expected_nonce=b"fresh")
    signed = verifier.extract_report_data(snp_report())
    assert verifier.verify_binding(evidence(), signed).vendor == "amd"
    with pytest.raises(AttestationError, match="nonce"):
        verifier.verify_binding(evidence().replace(nonce=b"stale"), signed)
    with pytest.raises(AttestationError, match="workload"):
        verifier.verify_binding(evidence().replace(workload_hash=b"tampered"), signed)


def test_snp_report_rejects_out_of_bounds_and_unbound_data() -> None:
    verifier = SNPGuestVerifier(expected_nonce=b"fresh")
    with pytest.raises(AttestationError, match="truncated"):
        verifier.extract_report_data(b"short")
    with pytest.raises(AttestationError, match="binding"):
        verifier.verify_binding(evidence(), bytes(64))


def test_tdx_report_data_is_read_from_the_signed_quote() -> None:
    quote = bytearray(1024)
    quote[960:1024] = signed_report_data()
    verifier = IntelDCAPVerifier(expected_nonce=b"fresh")
    assert (
        verifier.verify_binding(evidence(), verifier.extract_report_data(bytes(quote))).vendor
        == "amd"
    )
    with pytest.raises(AttestationError, match="truncated"):
        verifier.extract_report_data(b"short")
