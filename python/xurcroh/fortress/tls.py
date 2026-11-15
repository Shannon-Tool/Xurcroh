from __future__ import annotations

import ipaddress
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _private_bytes(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def generate_mutual_tls(output: Path, services: dict[str, list[str]]) -> None:
    if not services:
        raise ValueError("at least one Fortress service identity is required")
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Xurcroh deployment CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    (output / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.DER))
    for name, addresses in {**services, "client": ["localhost"]}.items():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        alternative_names: list[x509.GeneralName] = []
        for address in dict.fromkeys([name, *addresses, "localhost"]):
            try:
                alternative_names.append(x509.IPAddress(ipaddress.ip_address(address)))
            except ValueError:
                alternative_names.append(x509.DNSName(address))
        certificate = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName(alternative_names), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        _write_private(output / f"{name}.key", _private_bytes(key))
        (output / f"{name}.crt").write_bytes(certificate.public_bytes(serialization.Encoding.DER))
