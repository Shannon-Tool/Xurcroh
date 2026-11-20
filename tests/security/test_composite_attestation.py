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
    client.verify_components([component("amd"), component("nvidia")], nonce=b"n" * 32)
    assert verified == ["nvidia"]


def test_composite_verification_rejects_duplicate_vendor() -> None:
    client = RemoteAttestationClient(httpx.Client())
    with pytest.raises(AttestationError, match="duplicate"):
        client.verify_components([component("amd"), component("amd")], nonce=b"n" * 32)


def test_composite_verification_rejects_missing_required_vendor() -> None:
    client = RemoteAttestationClient(httpx.Client())
    with pytest.raises(AttestationError, match="required"):
        client.verify_components(
            [component("amd")],
            nonce=b"n" * 32,
            required_vendors={"amd", "nvidia"},
        )


def test_composite_verification_rejects_evidence_with_stale_self_declared_nonce() -> None:
    client = RemoteAttestationClient(
        httpx.Client(),
        hardware_verifier=lambda vendor, document, nonce: bytes.fromhex(
            str(document["report_data"])
        ),
    )
    stale = {**component("amd"), "nonce": (b"s" * 32).hex()}
    with pytest.raises(AttestationError, match="challenge nonce"):
        client.verify_components([stale, component("nvidia")], nonce=b"n" * 32)


def test_composite_verification_rejects_missing_nonce() -> None:
    client = RemoteAttestationClient(
        httpx.Client(),
        hardware_verifier=lambda vendor, document, nonce: bytes.fromhex(
            str(document["report_data"])
        ),
    )
    document = dict(component("amd"))
    del document["nonce"]
    with pytest.raises(AttestationError, match="challenge nonce"):
        client.verify_components([document, component("nvidia")], nonce=b"n" * 32)


def test_composite_nvidia_never_uses_unverified_local_report_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xurcroh.runtime.attestation import _verify_hardware

    document = component("nvidia")
    with pytest.raises(AttestationError, match="NRAS"):
        _verify_hardware("nvidia", document, b"n" * 32)

    verified = []
    monkeypatch.setattr(
        "xurcroh.runtime.attestation.NVIDIARemoteVerifier.verify",
        lambda self, evidence, document: verified.append(evidence.nonce),
    )
    client = RemoteAttestationClient(httpx.Client())
    client.verify_components([component("nvidia")], nonce=b"n" * 32)
    assert verified == [b"n" * 32]
