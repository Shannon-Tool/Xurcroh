from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest

from xurcroh.attestation import AttestationEvidence
from xurcroh.errors import AttestationError
from xurcroh.runtime.attestation import NVIDIARemoteVerifier


def evidence_and_report():
    nonce = b"n" * 32
    workload = b"workload"
    artifact = b"artifact"
    public_key = b"public-key"
    report_data = hashlib.sha256(nonce + workload + artifact + public_key).hexdigest() + "0" * 64
    evidence = AttestationEvidence(
        vendor="nvidia",
        nonce=nonce,
        report_data=report_data,
        workload_hash=workload,
        artifact_hash=artifact,
        public_key_hash=public_key,
    )
    report = base64.b64encode(
        json.dumps({"report_data": report_data, "signature": "signed"}).encode()
    ).decode()
    return nonce, evidence, {"report": report}


def test_nvidia_remote_verification_uses_signed_evidence_list() -> None:
    nonce, evidence, document = evidence_and_report()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"token": "signed-eat"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    NVIDIARemoteVerifier(client=client).verify(evidence, document)
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "https://nras.attestation.nvidia.com/v3/attest/gpu"
    payload = json.loads(request.content)
    assert payload["nonce"] == nonce.hex()
    assert payload["evidence_list"] == [{"evidence": document["report"], "type": "gpu"}]


def test_nvidia_remote_verification_rejects_missing_token() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    with pytest.raises(AttestationError, match="token"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)


def test_nvidia_remote_verification_rejects_nonce_mismatch() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"token": "signed-eat", "nonce": "00"})
        )
    )
    with pytest.raises(AttestationError, match="nonce"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)


def test_nvidia_remote_verification_rejects_transport_failure() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={}))
    )
    with pytest.raises(AttestationError, match="failed"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)
