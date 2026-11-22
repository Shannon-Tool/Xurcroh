"""The log, a witness and the hosted gateway as real processes over real HTTPS.

Everything a requester of the hosted service does is done here from outside: make a
request with an API key, fetch the trace and its log proof, have an independent
witness countersign the log, and check the result with ``xurcroh verify`` against the
live log.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import ssl
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

import httpx
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _certificate(directory: Path) -> tuple[Path, Path]:
    """A self-signed certificate for localhost, used as its own trust root."""

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "tls.crt", directory / "tls.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


def _run(arguments: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _signing_key(directory: Path, name: str, environment: dict[str, str]) -> tuple[Path, str]:
    key, public = directory / f"{name}.key", directory / f"{name}.pub"
    created = _run(
        ["xurcroh.signatures", str(key), "--generate", "--public-out", str(public)], environment
    )
    assert created.returncode == 0, created.stderr
    return key, public.read_bytes().hex()


def _wait(http: httpx.Client, url: str, process: subprocess.Popen[str], log: Path) -> None:
    deadline = time.monotonic() + 180
    while True:
        assert process.poll() is None, log.read_text()[-3000:]
        try:
            if http.get(url).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, log.read_text()[-3000:]
        time.sleep(0.25)


@pytest.mark.slow
def test_hosted_request_is_logged_witnessed_and_verified_from_outside(
    tiny_model_path: Path, tmp_path: Path
) -> None:
    certificate, tls_key = _certificate(tmp_path)
    environment = {
        **os.environ,
        "SSL_CERT_FILE": str(certificate),
        "XURCROH_LOG_WRITE_TOKEN": "write-token-for-this-test",
    }
    http = httpx.Client(verify=ssl.create_default_context(cafile=str(certificate)), timeout=120)
    log_key_file, log_key = _signing_key(tmp_path, "log", environment)
    witness_key_file, witness_key = _signing_key(tmp_path, "witness", environment)
    trace_key_file, trace_key = _signing_key(tmp_path, "gateway", environment)
    log_url = f"https://localhost:{_free_port()}"
    gateway_url = f"https://localhost:{_free_port()}"
    tls = ["--ssl-certfile", str(certificate), "--ssl-keyfile", str(tls_key)]

    api_key, other_key = "alice-secret-key", "bob-secret-key"
    (tmp_path / "keys.txt").write_text(
        f"alice:{hashlib.sha256(api_key.encode()).hexdigest()}\n"
        f"bob:{hashlib.sha256(other_key.encode()).hexdigest()}\n"
    )
    (tmp_path / "gateway.yaml").write_text(
        yaml.safe_dump(
            {
                "cluster": {"endpoint": "local://cpu"},
                "models": {"tiny": str(tiny_model_path)},
                "security": {
                    "trace_log_endpoint": log_url,
                    "trace_log_key": log_key,
                    "trace_log_token_env": "XURCROH_LOG_WRITE_TOKEN",
                },
            }
        )
    )
    processes: list[subprocess.Popen[str]] = []

    def start(name: str, arguments: list[str]) -> tuple[subprocess.Popen[str], Path]:
        output = tmp_path / f"{name}.log"
        process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", *arguments],
            env=environment,
            stdout=output.open("w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(process)
        return process, output

    try:
        log_process, log_output = start(
            "log",
            [
                "xurcroh.public_log",
                "serve",
                "--storage",
                str(tmp_path / "log-storage"),
                "--signing-key",
                str(log_key_file),
                "--host",
                "127.0.0.1",
                "--port",
                log_url.rsplit(":", 1)[1],
                *tls,
            ],
        )
        _wait(http, f"{log_url}/v1/public-key", log_process, log_output)
        gateway_process, gateway_output = start(
            "gateway",
            [
                "xurcroh.gateway",
                "--config",
                str(tmp_path / "gateway.yaml"),
                "--api-keys",
                str(tmp_path / "keys.txt"),
                "--trace-key",
                str(trace_key_file),
                "--host",
                "127.0.0.1",
                "--port",
                gateway_url.rsplit(":", 1)[1],
                *tls,
            ],
        )
        _wait(http, f"{gateway_url}/health", gateway_process, gateway_output)

        messages = [{"role": "user", "content": "hello"}]
        headers = {"Authorization": f"Bearer {api_key}"}
        response = http.post(
            f"{gateway_url}/v1/chat/completions",
            headers=headers,
            json={"model": "tiny", "messages": messages, "max_tokens": 2},
        )
        assert response.status_code == 200, response.text
        task = response.headers["X-Xurcroh-Task-Id"]
        output = response.json()["choices"][0]["message"]["content"]
        assert http.get(f"{gateway_url}/v1/trace-key").json()["public_key"] == trace_key

        # Another requester cannot read this task's trace; no key reads nothing.
        stranger = {"Authorization": f"Bearer {other_key}"}
        assert http.get(f"{gateway_url}/v1/tasks/{task}/trace", headers=stranger).status_code == 404
        assert http.get(f"{gateway_url}/v1/tasks/{task}/trace").status_code == 401

        # An independent witness checks the log and countersigns its head.
        witnessed = _run(
            [
                "xurcroh.public_log",
                "witness",
                "--url",
                log_url,
                "--log-key",
                log_key,
                "--signing-key",
                str(witness_key_file),
                "--state",
                str(tmp_path / "witness-state.json"),
            ],
            environment,
        )
        assert witnessed.returncode == 0, witnessed.stderr

        trace = http.get(f"{gateway_url}/v1/tasks/{task}/trace", headers=headers)
        proof = http.get(f"{gateway_url}/v1/tasks/{task}/proof", headers=headers)
        assert trace.status_code == 200 and proof.status_code == 200, proof.text
        (tmp_path / "trace.json").write_text(trace.text)
        (tmp_path / "proof.json").write_text(proof.text)
        (tmp_path / "messages.json").write_text(json.dumps(messages))
        (tmp_path / "output.txt").write_text(output)

        verify = [
            "xurcroh.cli",
            "verify",
            str(tmp_path / "trace.json"),
            "--public-key",
            trace_key,
            "--messages",
            str(tmp_path / "messages.json"),
            "--output",
            str(tmp_path / "output.txt"),
            "--proof",
            str(tmp_path / "proof.json"),
            "--log-key",
            log_key,
            "--log-url",
            log_url,
        ]
        checked = _run([*verify, "--witness-key", witness_key], environment)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "signature: verified" in checked.stdout
        assert "prompt: matches" in checked.stdout and "output: matches" in checked.stdout
        assert "public log: still holds that history" in checked.stdout
        assert "operator-asserted" in checked.stdout

        # A witness that never signed is not accepted, and a changed trace is rejected.
        unknown = _run([*verify, "--witness-key", log_key], environment)
        assert unknown.returncode == 1 and "witness" in unknown.stderr
        document = json.loads(trace.text)
        document["mode"] = "local"
        (tmp_path / "trace.json").write_text(json.dumps(document))
        assert _run(verify, environment).returncode == 1
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        http.close()
