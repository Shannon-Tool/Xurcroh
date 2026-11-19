from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from xurcroh.errors import AttestationError
from xurcroh.runtime.attestation import RemoteAttestationClient


def component(vendor: str) -> dict[str, object]:
    nonce = b"n" * 32
    workload = b"workload"
    artifact = b"artifact"
    public_key = b"public-key"
    report_data = hashlib.sha256(nonce + workload + artifact + public_key).hexdigest() + "0" * 64
    return {
        "vendor": vendor,
        "nonce": nonce.hex(),
        "report_data": report_data,
        "workload_hash": workload.hex(),
        "artifact_hash": artifact.hex(),
        "public_key_hash": public_key.hex(),
        "report": base64.b64encode(
            json_bytes := __import__("json")
            .dumps({"report_data": report_data, "signature": "signed"})
            .encode()
        ).decode(),
        "report_bytes": json_bytes,
    }


def test_composite_verification_requires_cpu_and_gpu_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = []
    monkeypatch.setattr(
        "xurcroh.runtime.attestation.NVIDIARemoteVerifier.verify",
        lambda self, evidence, document: verified.append("nvidia"),
    )
    client = RemoteAttestationClient(
        httpx.Client(),
        hardware_verifier=lambda vendor, document, nonce: bytes.fromhex(
            str(document["report_data"])
        ),
    )
    client.verify_components([component("amd"), component("nvidia")])
    assert verified == ["nvidia"]


def test_composite_verification_rejects_duplicate_vendor() -> None:
    client = RemoteAttestationClient(httpx.Client())
    with pytest.raises(AttestationError, match="duplicate"):
        client.verify_components([component("amd"), component("amd")])


def test_composite_verification_rejects_missing_required_vendor() -> None:
    client = RemoteAttestationClient(httpx.Client())
    with pytest.raises(AttestationError, match="required"):
        client.verify_components([component("amd")], required_vendors={"amd", "nvidia"})
