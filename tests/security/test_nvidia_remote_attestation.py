from __future__ import annotations

import base64
import hashlib
import json
import time

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

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


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def eat_token(key, nonce: bytes) -> str:
    header = b64url(json.dumps({"alg": "ES256", "typ": "JWT"}).encode())
    payload = b64url(json.dumps({"eat_nonce": nonce.hex(), "iat": int(time.time())}).encode())
    der = key.sign(f"{header}.{payload}".encode(), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{header}.{payload}.{b64url(signature)}"


def test_nvidia_remote_verification_uses_signed_evidence_list() -> None:
    nonce, evidence, document = evidence_and_report()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"verified": True, "token": "signed-eat"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    NVIDIARemoteVerifier(client=client).verify(evidence, document)
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "https://nras.attestation.nvidia.com/v3/attest/gpu"
    payload = json.loads(request.content)
    assert payload["nonce"] == nonce.hex()
    assert payload["evidence_list"] == [{"evidence": document["report"], "type": "gpu"}]


def test_nvidia_remote_verification_rejects_missing_verified_decision() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"token": "signed-eat"})
        )
    )
    with pytest.raises(AttestationError, match="failed"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)


def test_nvidia_remote_verification_rejects_negative_verified_decision() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"verified": False, "token": "eat"})
        )
    )
    with pytest.raises(AttestationError, match="failed"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)


def test_nvidia_remote_verification_rejects_missing_token() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"verified": True}))
    )
    with pytest.raises(AttestationError, match="token"):
        NVIDIARemoteVerifier(client=client).verify(evidence, document)


def test_nvidia_remote_verification_rejects_nonce_mismatch() -> None:
    _, evidence, document = evidence_and_report()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"verified": True, "token": "signed-eat", "nonce": "00"}
            )
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


def test_nvidia_remote_verification_validates_token_with_pinned_key() -> None:
    nonce, evidence, document = evidence_and_report()
    key = ec.generate_private_key(ec.SECP256R1())
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"verified": True, "token": eat_token(key, nonce)})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    NVIDIARemoteVerifier(client=client, verifier_public_key_pem=public_key).verify(
        evidence, document
    )

    def bad_token_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"verified": True, "token": "not-a-jwt"})

    client = httpx.Client(transport=httpx.MockTransport(bad_token_handler))
    with pytest.raises(AttestationError, match="invalid"):
        NVIDIARemoteVerifier(client=client, verifier_public_key_pem=public_key).verify(
            evidence, document
        )

    other = ec.generate_private_key(ec.SECP256R1())

    def wrong_key_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"verified": True, "token": eat_token(other, nonce)})

    client = httpx.Client(transport=httpx.MockTransport(wrong_key_handler))
    with pytest.raises(AttestationError, match="invalid"):
        NVIDIARemoteVerifier(client=client, verifier_public_key_pem=public_key).verify(
            evidence, document
        )


def test_nvidia_remote_verification_rejects_token_with_wrong_nonce() -> None:
    _, evidence, document = evidence_and_report()
    key = ec.generate_private_key(ec.SECP256R1())
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"verified": True, "token": eat_token(key, b"x" * 32)}
            )
        )
    )
    with pytest.raises(AttestationError, match="nonce"):
        NVIDIARemoteVerifier(client=client, verifier_public_key_pem=public_key).verify(
            evidence, document
        )
