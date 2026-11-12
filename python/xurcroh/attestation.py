from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from xurcroh.errors import AttestationError, CompatibilityError


@dataclass(frozen=True)
class AttestationEvidence:
    vendor: str
    nonce: bytes
    report_data: str
    workload_hash: bytes
    artifact_hash: bytes
    public_key_hash: bytes

    def replace(self, **changes: Any) -> AttestationEvidence:
        return replace(self, **changes)


class AttestationVerifier:
    def __init__(self, *, expected_nonce: bytes) -> None:
        self.expected_nonce = expected_nonce

    def verify_binding(
        self, evidence: AttestationEvidence, signed_report_data: bytes | None = None
    ) -> AttestationEvidence:
        if signed_report_data is None:
            raise AttestationError("hardware-signed report data is required")
        if not hmac.compare_digest(evidence.nonce, self.expected_nonce):
            raise AttestationError("attestation nonce is not fresh or does not match")
        expected = hashlib.sha256(
            evidence.nonce
            + evidence.workload_hash
            + evidence.artifact_hash
            + evidence.public_key_hash
        ).digest()
        if (
            len(signed_report_data) != 64
            or not hmac.compare_digest(signed_report_data[:32], expected)
            or any(signed_report_data[32:])
        ):
            raise AttestationError(
                "attestation workload, artifact, or public-key binding is invalid"
            )
        return evidence


class SNPGuestVerifier(AttestationVerifier):
    @staticmethod
    def extract_report_data(report: bytes | str | Path) -> bytes:
        raw = Path(report).read_bytes() if isinstance(report, str | Path) else report
        if len(raw) < 0x90:
            raise AttestationError("SEV-SNP report is truncated")
        return raw[0x50:0x90]

    def verify(
        self, evidence: AttestationEvidence, *, certificates: str | Path, report: str | Path
    ) -> None:
        self.verify_binding(evidence, self.extract_report_data(report))
        executable = shutil.which("snpguest")
        if executable is None:
            raise CompatibilityError("snpguest is required for SEV-SNP verification")
        result = subprocess.run(  # noqa: S603 - resolved vendor tool, no shell
            [executable, "verify", "attestation", str(certificates), str(report)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AttestationError(f"SEV-SNP verification failed: {result.stderr.strip()}")


class NVIDIAAttestationVerifier(AttestationVerifier):
    @staticmethod
    def extract_report_data(report: bytes) -> bytes:
        try:
            document = json.loads(report)
            signed = bytes.fromhex(document["report_data"])
            signature = document["signature"]
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise AttestationError("NVIDIA signed evidence is malformed") from error
        if not isinstance(signature, str) or not signature:
            raise AttestationError("NVIDIA evidence is not signed")
        return signed

    def verify(self, evidence: AttestationEvidence, *, report: bytes) -> None:
        self.verify_binding(evidence, self.extract_report_data(report))
        result = subprocess.run(  # noqa: S603 - current Python executable, no shell
            [sys.executable, "-m", "verifier.cc_admin", "--nonce", evidence.nonce.hex()],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + result.stderr).lower()
        if result.returncode != 0 or "devtools" in output or "fail" in output:
            raise AttestationError(
                "NVIDIA confidential GPU attestation failed or devtools is enabled"
            )


@dataclass(frozen=True)
class DCAPCollateral:
    trusted_root_ca: str
    pck_certificate: str
    pck_signing_chain: str
    root_ca_crl: str
    intermediate_ca_crl: str
    tcb_info: str
    tcb_signing_chain: str
    qe_identity: str
    expiration_date: datetime


class IntelDCAPVerifier(AttestationVerifier):
    @staticmethod
    def extract_report_data(quote: bytes) -> bytes:
        if len(quote) < 1024:
            raise AttestationError("Intel TDX quote is truncated")
        return quote[960:1024]

    def verify(
        self,
        evidence: AttestationEvidence,
        *,
        quote: bytes,
        collateral: DCAPCollateral,
    ) -> Any:
        self.verify_binding(evidence, self.extract_report_data(quote))
        try:
            import sgx_dcap_quote_verify
        except ImportError as error:
            raise CompatibilityError(
                "Intel DCAP verification requires sgx-dcap-quote-verify-python"
            ) from error
        result = sgx_dcap_quote_verify.verify(
            collateral.trusted_root_ca,
            collateral.pck_certificate,
            collateral.pck_signing_chain,
            collateral.root_ca_crl,
            collateral.intermediate_ca_crl,
            collateral.tcb_info,
            collateral.tcb_signing_chain,
            quote,
            collateral.qe_identity,
            collateral.expiration_date,
        )
        if not result.ok:
            raise AttestationError("Intel DCAP quote verification failed")
        return result


@dataclass(frozen=True)
class AttestationPolicy:
    mode: str
    required_vendors: tuple[str, ...] = ()

    def validate(self, evidence: tuple[AttestationEvidence, ...]) -> None:
        if self.mode == "tee" and not evidence:
            raise AttestationError("TEE mode requires attestation evidence")
        supported = {"amd", "intel", "nvidia"}
        if any(item.vendor not in supported for item in evidence):
            raise AttestationError("unsupported attestation vendor")
        if self.required_vendors and {item.vendor for item in evidence} != set(
            self.required_vendors
        ):
            raise AttestationError("attestation evidence does not match required vendors")
        if self.mode not in {"strict", "tee", "non_tee", "development"}:
            raise AttestationError(f"unsupported attestation mode: {self.mode}")
