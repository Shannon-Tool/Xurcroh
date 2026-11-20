from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import httpx
import pytest

from integration.test_stage_sdk import running_stage
from shards.identities import StageIdentities
from Xurcroh import Xurcroh
from xurcroh.errors import AttestationError
from xurcroh.runtime.attestation import RemoteAttestationClient


def attestation_document(public_key: bytes) -> dict[str, object]:
    workload = b"workload"
    artifact = b"artifact"
    return {
        "vendor": "amd",
        "workload_hash": workload.hex(),
        "artifact_hash": artifact.hex(),
        "public_key_hash": public_key.hex(),
        "report": base64.b64encode(b"report").decode(),
        "certificates": base64.b64encode(b"certificates").decode(),
    }


def compiled_manifest(tiny_llama_path: Path, tmp_path: Path):
    identities = StageIdentities.create(1)
    result = identities.compile(tiny_llama_path, tmp_path / "compiled", seed=241)
    return identities, result


def test_remote_attestation_binds_fresh_challenge_and_rejects_tampering(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    identities, compiled = compiled_manifest(tiny_llama_path, tmp_path)
    document = attestation_document(identities.recipients[0].public_key)
    verified_nonces = []

    def verify_hardware(vendor, document, nonce):
        assert vendor == "amd"
        verified_nonces.append(nonce)
        return bytes.fromhex(str(document["report_data"])) + bytes(32)

    with running_stage(compiled.manifest_path, 0, identities.private_keys[0], document) as endpoint:
        client = RemoteAttestationClient(httpx.Client(), hardware_verifier=verify_hardware)
        client.verify_worker(endpoint)
        client.verify_worker(endpoint)
        assert len(set(verified_nonces)) == 2
        client.close()
    with running_stage(
        compiled.manifest_path,
        0,
        identities.private_keys[0],
        {**document, "report_data": "0" * 64},
    ) as endpoint:
        client = RemoteAttestationClient(httpx.Client(), hardware_verifier=verify_hardware)
        with pytest.raises(AttestationError, match="binding"):
            client.verify_worker(endpoint)
        client.close()


def test_tee_stage_sdk_exchanges_and_verifies_evidence(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities, compiled = compiled_manifest(tiny_llama_path, tmp_path)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    monkeypatch.setattr(
        "xurcroh.runtime.attestation._verify_hardware",
        lambda vendor, document, nonce: bytes.fromhex(str(document["report_data"])) + bytes(32),
    )
    with (
        running_stage(
            compiled.manifest_path,
            0,
            identities.private_keys[0],
            attestation_document(identities.recipients[0].public_key),
        ) as endpoint,
        Xurcroh(
            cluster={"name": "stage-0", "endpoint": endpoint},
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "manifest_path": str(compiled.manifest_path),
                    "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                }
            },
            security={"mode": "development", "require_attestation": True},
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
            temperature=0,
        )
    assert response.usage.completion_tokens == 1


def amd_document() -> dict[str, object]:
    nonce = b"f" * 32
    workload = b"workload"
    artifact = b"artifact"
    public_key = b"public-key"
    report = bytearray(0x2A0)
    report[0x50:0x90] = hashlib.sha256(nonce + workload + artifact + public_key).digest() + bytes(
        32
    )
    return {
        "vendor": "amd",
        "nonce": nonce.hex(),
        "report_data": hashlib.sha256(nonce + workload + artifact + public_key).hexdigest()
        + "0" * 64,
        "workload_hash": workload.hex(),
        "artifact_hash": artifact.hex(),
        "public_key_hash": public_key.hex(),
        "report": base64.b64encode(bytes(report)).decode(),
        "certificates": base64.b64encode(b"certificates").decode(),
    }


def test_amd_hardware_path_extracts_report_data_before_tempdir_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xurcroh.runtime.attestation import _verify_hardware

    monkeypatch.setattr(
        "xurcroh.runtime.attestation.SNPGuestVerifier.verify", lambda *values, **kwargs: None
    )
    document = amd_document()
    signed = _verify_hardware("amd", document, bytes.fromhex(str(document["nonce"])))
    assert signed[:32] == bytes.fromhex(str(document["report_data"]))[:32]


def test_amd_hardware_path_converts_extraction_failure_to_attestation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xurcroh.runtime.attestation import _verify_hardware

    monkeypatch.setattr(
        "xurcroh.runtime.attestation.SNPGuestVerifier.verify", lambda *values, **kwargs: None
    )

    def broken_extract(report):
        raise OSError("report vanished")

    monkeypatch.setattr(
        "xurcroh.runtime.attestation.SNPGuestVerifier.extract_report_data", broken_extract
    )
    with pytest.raises(AttestationError, match="extract"):
        _verify_hardware("amd", amd_document(), b"f" * 32)


def test_worker_rejects_composite_evidence_not_bound_to_challenge_nonce() -> None:
    challenge = b"c" * 32
    component = amd_document()
    component["nonce"] = (b"s" * 32).hex()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/challenge"):
            return httpx.Response(200, json={"nonce": challenge.hex()})
        return httpx.Response(200, json={"components": [component]})

    client = RemoteAttestationClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        hardware_verifier=lambda vendor, document, nonce: bytes.fromhex(
            str(document["report_data"])
        ),
    )
    with pytest.raises(AttestationError, match="challenge nonce"):
        client.verify_worker("https://worker.example")


def test_worker_rejects_single_evidence_with_missing_or_stale_nonce() -> None:
    challenge = b"c" * 32

    def make_client(document: dict[str, object]) -> RemoteAttestationClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/challenge"):
                return httpx.Response(200, json={"nonce": challenge.hex()})
            return httpx.Response(200, json=document)

        return RemoteAttestationClient(
            httpx.Client(transport=httpx.MockTransport(handler)),
            hardware_verifier=lambda vendor, document, nonce: bytes.fromhex(
                str(document["report_data"])
            ),
        )

    document = amd_document()
    del document["nonce"]
    with pytest.raises(AttestationError, match="challenge nonce"):
        make_client(document).verify_worker("https://worker.example")
    stale = amd_document()
    stale["nonce"] = (b"s" * 32).hex()
    with pytest.raises(AttestationError, match="challenge nonce"):
        make_client(stale).verify_worker("https://worker.example")
