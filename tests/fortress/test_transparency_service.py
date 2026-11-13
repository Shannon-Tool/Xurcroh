from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from xurcroh.transparency import TransparencyError, verify_transparency_log
from xurcroh.transparency_cli import main as transparency_main
from xurcroh.transparency_service import TransparencyService


def test_live_transparency_service_is_persistent_and_authenticated(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "public-log.json"
    service = TransparencyService(path, signing_key=key, write_token="write-secret")
    client = TestClient(service.app)
    unauthorized = client.post(
        "/v1/entries", json={"image_digest": "a" * 64, "measurement": "b" * 64}
    )
    assert unauthorized.status_code == 401
    response = client.post(
        "/v1/entries",
        json={"image_digest": "a" * 64, "measurement": "b" * 64},
        headers={"Authorization": "Bearer write-secret"},
    )
    assert response.status_code == 201
    public = client.get("/v1/log")
    verify_transparency_log(
        public.json(),
        public_key=key.public_key().public_bytes_raw(),
        expected_measurements={"a" * 64: "b" * 64},
    )
    restarted = TransparencyService(path, signing_key=key, write_token="write-secret")
    with TestClient(restarted.app) as restarted_client:
        latest = restarted_client.get(f"/v1/measurements/{'a' * 64}")
        assert latest.json()["measurement"] == "b" * 64


def test_transparency_service_detects_disk_tampering(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "public-log.json"
    service = TransparencyService(path, signing_key=key, write_token="write-secret")
    client = TestClient(service.app)
    client.post(
        "/v1/entries",
        json={"image_digest": "a" * 64, "measurement": "b" * 64},
        headers={"Authorization": "Bearer write-secret"},
    )
    document = json.loads(path.read_text())
    document["entries"][0]["measurement"] = "c" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(TransparencyError, match="signature"):
        client.get("/v1/log")


def test_transparency_cli_uses_indirect_secret_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        transparency_main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--write-token-env" in output
    assert "--signing-key" in output
