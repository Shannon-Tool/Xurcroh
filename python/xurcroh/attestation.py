from __future__ import annotations

import hashlib
import hmac
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

    def verify_binding(self, evidence: AttestationEvidence) -> AttestationEvidence:
        if not hmac.compare_digest(evidence.nonce, self.expected_nonce):
            raise AttestationError("attestation nonce is not fresh or does not match")
        expected = hashlib.sha256(
            evidence.nonce
            + evidence.workload_hash
            + evidence.artifact_hash
            + evidence.public_key_hash
        ).hexdigest()
        if not hmac.compare_digest(evidence.report_data, expected):
            raise AttestationError(
                "attestation workload, artifact, or public-key binding is invalid"
            )
        return evidence


class SNPGuestVerifier(AttestationVerifier):
    def verify(
        self, evidence: AttestationEvidence, *, certificates: str | Path, report: str | Path
    ) -> None:
        self.verify_binding(evidence)
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
    def verify(self, evidence: AttestationEvidence) -> None:
        self.verify_binding(evidence)
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
    def verify(
        self,
        evidence: AttestationEvidence,
        *,
        quote: bytes,
        collateral: DCAPCollateral,
    ) -> Any:
        self.verify_binding(evidence)
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
