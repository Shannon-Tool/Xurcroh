#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def private_bytes(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def public_bytes(certificate: x509.Certificate) -> bytes:
    return certificate.public_bytes(serialization.Encoding.DER)


def generate(output: Path, services: dict[str, list[str]]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Xurcroh validation CA")])
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
    write_private(output / "ca.key", private_bytes(ca_key))
    (output / "ca.crt").write_bytes(public_bytes(ca))
    for name, addresses in {**services, "client": ["localhost"]}.items():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        alternative_names = []
        for address in addresses:
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
        write_private(output / f"{name}.key", private_bytes(key))
        (output / f"{name}.crt").write_bytes(public_bytes(certificate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", action="append", required=True, metavar="NAME=HOST,HOST")
    args = parser.parse_args()
    services = {}
    for value in args.service:
        name, separator, addresses = value.partition("=")
        if not separator or not name or not addresses:
            parser.error("--service requires NAME=HOST,HOST")
        services[name] = addresses.split(",")
    generate(args.output, services)


if __name__ == "__main__":
    main()
