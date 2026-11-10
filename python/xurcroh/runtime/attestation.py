from __future__ import annotations

import base64
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from xurcroh.attestation import (
    AttestationEvidence,
    AttestationVerifier,
    DCAPCollateral,
    IntelDCAPVerifier,
    SNPGuestVerifier,
)
from xurcroh.errors import AttestationError, CompatibilityError


def _verify_hardware(vendor: str, document: dict[str, Any], nonce: bytes) -> None:
    evidence = AttestationEvidence(
        vendor=vendor,
        nonce=nonce,
        report_data=str(document["report_data"]),
        workload_hash=bytes.fromhex(document["workload_hash"]),
        artifact_hash=bytes.fromhex(document["artifact_hash"]),
        public_key_hash=bytes.fromhex(document["public_key_hash"]),
    )
    if vendor == "amd":
        with tempfile.TemporaryDirectory(prefix="xurcroh-attestation-") as directory:
            root = Path(directory)
            report = root / "report.bin"
            certificates = root / "certificates.bin"
            report.write_bytes(base64.b64decode(document["report"], validate=True))
            certificates.write_bytes(base64.b64decode(document["certificates"], validate=True))
            SNPGuestVerifier(expected_nonce=nonce).verify(
                evidence, certificates=certificates, report=report
            )
        return
    if vendor == "intel":
        collateral_data = document["collateral"]
        collateral = DCAPCollateral(
            trusted_root_ca=collateral_data["trusted_root_ca"],
            pck_certificate=collateral_data["pck_certificate"],
            pck_signing_chain=collateral_data["pck_signing_chain"],
            root_ca_crl=collateral_data["root_ca_crl"],
            intermediate_ca_crl=collateral_data["intermediate_ca_crl"],
            tcb_info=collateral_data["tcb_info"],
            tcb_signing_chain=collateral_data["tcb_signing_chain"],
            qe_identity=collateral_data["qe_identity"],
            expiration_date=datetime.fromisoformat(collateral_data["expiration_date"]),
        )
        IntelDCAPVerifier(expected_nonce=nonce).verify(
            evidence,
            quote=base64.b64decode(document["quote"], validate=True),
            collateral=collateral,
        )
        return
    if vendor == "nvidia":
        raise CompatibilityError(
            "remote NVIDIA evidence requires the vendor remote-attestation transport; "
            "the local cc_admin adapter cannot verify a worker report"
        )
    raise AttestationError("unsupported attestation vendor")


class RemoteAttestationClient:
    def __init__(
        self,
        client: httpx.Client,
        *,
        hardware_verifier: Callable[[str, dict[str, Any], bytes], None] | None = None,
    ) -> None:
        self._client = client
        self._hardware_verifier = hardware_verifier or _verify_hardware

    @staticmethod
    def _endpoint(endpoint: str) -> str:
        if endpoint.startswith("http+dev://"):
            return "http://" + endpoint.removeprefix("http+dev://")
        return endpoint.rstrip("/")

    def verify_worker(self, endpoint: str) -> None:
        base = self._endpoint(endpoint)
        try:
            challenge_response = self._client.get(f"{base}/v1/attestation/challenge")
            challenge_response.raise_for_status()
            nonce = bytes.fromhex(challenge_response.json()["nonce"])
            evidence_response = self._client.get(
                f"{base}/v1/attestation/evidence", params={"nonce": nonce.hex()}
            )
            evidence_response.raise_for_status()
            document = evidence_response.json()
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise AttestationError("worker did not provide attestation evidence") from error
        vendor = str(document.get("vendor", ""))
        evidence_nonce = bytes.fromhex(document.get("nonce", nonce.hex()))
        evidence = AttestationEvidence(
            vendor=vendor,
            nonce=evidence_nonce,
            report_data=str(document["report_data"]),
            workload_hash=bytes.fromhex(document["workload_hash"]),
            artifact_hash=bytes.fromhex(document["artifact_hash"]),
            public_key_hash=bytes.fromhex(document["public_key_hash"]),
        )
        AttestationVerifier(expected_nonce=nonce).verify_binding(evidence)
        self._hardware_verifier(vendor, document, nonce)

    def close(self) -> None:
        return None
