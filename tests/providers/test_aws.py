from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import cbor2
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.x509.oid import NameOID

from xurcroh.errors import ProviderError
from xurcroh.providers.aws import AwsNitroClient


class FakeEC2:
    def __init__(self) -> None:
        self.calls = []

    def describe_instance_types(self, **values):
        self.calls.append(values)
        return {
            "InstanceTypes": [
                {
                    "InstanceType": "m5.xlarge",
                    "EnclaveOptions": {"Enabled": True},
                    "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
                }
            ]
        }

    def run_instances(self, **values):
        self.calls.append(values)
        return {"Instances": [{"InstanceId": "i-test"}]}

    def describe_instances(self, **values):
        self.calls.append(values)
        return {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-test", "State": {"Name": "running"}}]}
            ]
        }

    def terminate_instances(self, **values):
        self.calls.append(values)
        return {}


class FakeKMS:
    def __init__(self) -> None:
        self.calls = []

    def decrypt(self, **values):
        self.calls.append(values)
        return {"KeyId": "key-test", "Plaintext": b"protected secret"}


class FakeCostExplorer:
    def get_cost_and_usage_with_resources(self, **values):
        assert "NextPageToken" not in values or values["NextPageToken"] is not None
        return {
            "ResultsByTime": [
                {
                    "Groups": [
                        {
                            "Keys": ["i-test"],
                            "Metrics": {"UnblendedCost": {"Amount": "1.75", "Unit": "USD"}},
                        }
                    ]
                }
            ]
        }


def client(ec2=None, kms=None, root_certificate=None):
    return AwsNitroClient(
        region_id="us-east-1",
        ec2_client=ec2 or FakeEC2(),
        kms_client=kms or FakeKMS(),
        cost_client=FakeCostExplorer(),
        nitro_root_certificate=root_certificate,
    )


def test_aws_nitro_parent_lifecycle_and_costs() -> None:
    ec2 = FakeEC2()
    aws = client(ec2)
    offer = aws.offers()[0]
    assert offer["tee_type"] == "nitro"
    assert offer["gpu_memory_bytes"] == 0
    assert aws.offers("H100") == []
    instance = aws.create(
        offer,
        label="tee-parent",
        image_id="ami-test",
        key_name="tee-key",
        subnet_id="subnet-test",
        security_group_ids=["sg-test"],
    )
    assert instance == "i-test"
    run = ec2.calls[-1]
    assert run["EnclaveOptions"] == {"Enabled": True}
    assert run["MetadataOptions"]["HttpTokens"] == "required"
    assert run["HibernationOptions"] == {"Configured": False}
    assert aws.remaining(instance) is True
    assert aws.charges(instance) == 1.75
    aws.destroy(instance)
    assert ec2.calls[-1] == {"InstanceIds": [instance]}


def certificate_pair():
    now = datetime.now(UTC)
    root_key = ec.generate_private_key(ec.SECP384R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Nitro Root")])
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .sign(root_key, hashes.SHA384())
    )
    leaf_key = ec.generate_private_key(ec.SECP384R1())
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Nitro Enclave")])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(root_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(root_key, hashes.SHA384())
    )
    return root, leaf_key, leaf


def attestation_document(root, leaf_key, leaf, pcrs, nonce):
    protected = cbor2.dumps({1: -35})
    document = {
        "module_id": "test-nitro-module",
        "timestamp": int(time.time() * 1000),
        "digest": "SHA384",
        "pcrs": pcrs,
        "certificate": leaf.public_bytes(serialization.Encoding.DER),
        "cabundle": [root.public_bytes(serialization.Encoding.DER)],
        "public_key": b"session public key",
        "nonce": nonce,
    }
    payload = cbor2.dumps(document)
    sig_structure = cbor2.dumps(["Signature1", protected, b"", payload])
    der = leaf_key.sign(sig_structure, ec.ECDSA(hashes.SHA384()))
    r, s = decode_dss_signature(der)
    signature = r.to_bytes(48, "big") + s.to_bytes(48, "big")
    return cbor2.dumps(cbor2.CBORTag(18, [protected, {}, payload, signature]))


def verified_document():
    root, leaf_key, leaf = certificate_pair()
    pcrs = {index: bytes([index]) * 48 for index in range(3)}
    nonce = b"fresh nonce"
    document = attestation_document(root, leaf_key, leaf, pcrs, nonce)
    root_pem = root.public_bytes(serialization.Encoding.PEM)
    return document, root_pem, pcrs, nonce


def test_aws_attestation_verifies_cose_chain_pcr_and_nonce() -> None:
    document, root_pem, pcrs, nonce = verified_document()
    aws = client(root_certificate=root_pem)
    claims = aws.verify_attestation_document(
        document,
        expected_pcrs=pcrs,
        expected_nonce=nonce,
        expected_public_key=b"session public key",
    )
    assert claims["digest"] == "SHA384"


def test_aws_attestation_rejects_wrong_measurement() -> None:
    document, root_pem, pcrs, nonce = verified_document()
    aws = client(root_certificate=root_pem)
    expected = dict(pcrs)
    expected[0] = b"\xff" * 48
    with pytest.raises(ProviderError, match="PCR"):
        aws.verify_attestation_document(
            document,
            expected_pcrs=expected,
            expected_nonce=nonce,
            expected_public_key=b"session public key",
        )


def test_aws_attestation_rejects_non_bytes_cose_signature() -> None:
    root, leaf_key, leaf = certificate_pair()
    pcrs = {index: bytes([index]) * 48 for index in range(3)}
    nonce = b"fresh nonce"
    document = attestation_document(root, leaf_key, leaf, pcrs, nonce)
    cose = cbor2.loads(document)
    tampered = cbor2.dumps(
        cbor2.CBORTag(18, [cose.value[0], cose.value[1], cose.value[2], "a" * 96])
    )
    aws = client(root_certificate=root.public_bytes(serialization.Encoding.PEM))
    with pytest.raises(ProviderError):
        aws.verify_attestation_document(
            tampered,
            expected_pcrs=pcrs,
            expected_nonce=nonce,
            expected_public_key=b"session public key",
        )


def test_aws_kms_decrypt_requires_valid_attestation() -> None:
    document, root_pem, pcrs, nonce = verified_document()
    kms = FakeKMS()
    aws = client(kms=kms, root_certificate=root_pem)
    plaintext = aws.decrypt_with_attestation(
        key_id="key-test",
        ciphertext=b"ciphertext",
        attestation_document=document,
        expected_pcrs=pcrs,
        expected_nonce=nonce,
        expected_public_key=b"session public key",
    )
    assert plaintext == b"protected secret"
    assert kms.calls[0]["Recipient"]["AttestationDocument"] == document


def test_aws_enclave_command_never_enables_debug_mode() -> None:
    command = AwsNitroClient.build_enclave_command(
        eif_path="/srv/xurcroh/worker.eif",
        cpu_count=2,
        memory_mib=4096,
        enclave_cid=16,
    )
    assert command[:2] == ["nitro-cli", "run-enclave"]
    assert "--debug-mode" not in command
    with pytest.raises(ValueError, match="debug"):
        AwsNitroClient.build_enclave_command(
            eif_path="/srv/xurcroh/worker.eif",
            cpu_count=2,
            memory_mib=4096,
            enclave_cid=16,
            debug_mode=True,
        )
