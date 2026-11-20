from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from xurcroh.attestation import (
    INTEL_SGX_ROOT_CA_SPKI_SHA256,
    AttestationEvidence,
    AttestationVerifier,
    DCAPCollateral,
    IntelDCAPVerifier,
    NVIDIAAttestationVerifier,
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


def root_ca_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Intel SGX Root CA")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime(2020, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime(2040, 1, 1, tzinfo=UTC))
        .sign(key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def dcap_collateral(trusted_root_ca: str) -> DCAPCollateral:
    return DCAPCollateral(
        trusted_root_ca=trusted_root_ca,
        pck_certificate="pck",
        pck_signing_chain="pck-chain",
        root_ca_crl="root-crl",
        intermediate_ca_crl="intermediate-crl",
        tcb_info="tcb-info",
        tcb_signing_chain="tcb-chain",
        qe_identity="qe-identity",
        expiration_date=datetime(2030, 1, 1, tzinfo=UTC),
    )


def tdx_quote() -> bytes:
    quote = bytearray(1024)
    quote[960:1024] = signed_report_data()
    return bytes(quote)


def test_dcap_collateral_root_ca_must_match_pinned_intel_root(monkeypatch) -> None:
    pem = root_ca_pem()
    certificate = x509.load_pem_x509_certificate(pem.encode("ascii"))
    spki = certificate.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    verifier = IntelDCAPVerifier(
        expected_nonce=b"fresh", expected_root_ca_spki_sha256=hashlib.sha256(spki).digest()
    )
    calls = []

    def fake_verify(*values):
        calls.append(values)
        return SimpleNamespace(ok=True)

    monkeypatch.setitem(sys.modules, "sgx_dcap_quote_verify", SimpleNamespace(verify=fake_verify))
    result = verifier.verify(
        evidence(),
        quote=tdx_quote(),
        collateral=dcap_collateral(pem),
    )
    assert result.ok is True
    assert len(calls) == 1
    with pytest.raises(AttestationError, match="pinned Intel"):
        verifier.verify(
            evidence(),
            quote=tdx_quote(),
            collateral=dcap_collateral(root_ca_pem()),
        )
    assert len(calls) == 1


def test_dcap_collateral_rejects_unparseable_root_ca() -> None:
    verifier = IntelDCAPVerifier(expected_nonce=b"fresh")
    with pytest.raises(AttestationError, match="PEM"):
        verifier.verify(
            evidence(),
            quote=tdx_quote(),
            collateral=dcap_collateral("not a certificate"),
        )


def test_dcap_pin_defaults_to_well_known_intel_root_spki() -> None:
    assert len(INTEL_SGX_ROOT_CA_SPKI_SHA256) == 32
    verifier = IntelDCAPVerifier(expected_nonce=b"fresh")
    with pytest.raises(AttestationError, match="pinned Intel"):
        verifier._verify_collateral_root(dcap_collateral(root_ca_pem()))


def test_nvidia_local_verifier_fails_closed_without_hardware_signature() -> None:
    nonce = b"fresh"
    signed = signed_report_data()
    report = (
        __import__("json")
        .dumps({"report_data": signed.hex(), "signature": "attester-typed"})
        .encode()
    )
    verifier = NVIDIAAttestationVerifier(expected_nonce=nonce)
    with pytest.raises(AttestationError, match="remote NRAS"):
        verifier.verify(evidence().replace(vendor="nvidia"), report=report)
