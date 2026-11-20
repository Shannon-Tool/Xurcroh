from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from xurcroh.errors import ProviderError
from xurcroh.providers.oci import OciConfidentialClient


def response(data):
    return SimpleNamespace(data=data)


class FakeCompute:
    def __init__(self) -> None:
        self.requests = []
        self.instance = SimpleNamespace(
            id="ocid1.instance.oc1..test",
            lifecycle_state="RUNNING",
            freeform_tags={"xurcroh-instance": "tee-worker"},
        )

    def list_shapes(self, **values):
        self.requests.append(values)
        return response(
            [
                SimpleNamespace(shape="VM.Standard.E5.Flex", gpus=0, processor_description="AMD"),
                SimpleNamespace(shape="VM.Standard.E4.Flex", gpus=0, processor_description="AMD"),
                SimpleNamespace(shape="VM.GPU.A10.1", gpus=1, processor_description="AMD"),
            ]
        )

    def launch_instance(self, details):
        self.requests.append(details)
        return response(self.instance)

    def get_instance(self, instance_id):
        self.requests.append(instance_id)
        return response(self.instance)

    def terminate_instance(self, instance_id):
        self.requests.append(instance_id)
        return response(None)


class FakeUsage:
    def __init__(self) -> None:
        self.requests = []

    def request_summarized_usages(self, details):
        self.requests.append(details)
        return response(
            SimpleNamespace(
                items=[
                    SimpleNamespace(
                        resource_id="ocid1.instance.oc1..test",
                        computed_amount=1.5,
                        currency="USD",
                    )
                ]
            )
        )


def client(compute=None, usage=None, **values):
    return OciConfidentialClient(
        tenancy_id="ocid1.tenancy.oc1..test",
        compartment_id="ocid1.compartment.oc1..test",
        compute_client=compute or FakeCompute(),
        usage_client=usage or FakeUsage(),
        **values,
    )


def test_oci_confidential_cpu_lifecycle() -> None:
    compute = FakeCompute()
    usage = FakeUsage()
    oci = client(compute, usage)
    offer = oci.offers()[0]
    assert offer["id"] == "VM.Standard.E5.Flex"
    assert offer["tee_type"] == "sev-snp"
    assert offer["gpu_memory_bytes"] == 0
    assert len(oci.offers()) == 1
    instance = oci.create(
        offer,
        label="tee-worker",
        image_id="ocid1.image.oc1..test",
        availability_domain="PHX-AD-1",
        subnet_id="ocid1.subnet.oc1..test",
        network_security_group_ids=["ocid1.nsg.oc1..test"],
        ssh_public_key="ssh-ed25519 AAA",
        ocpus=4,
        memory_gb=64,
    )
    assert instance == "ocid1.instance.oc1..test"
    launch = next(value for value in compute.requests if hasattr(value, "platform_config"))
    assert launch.platform_config.is_memory_encryption_enabled is True
    assert launch.platform_config.type == "AMD_VM"
    assert launch.create_vnic_details.nsg_ids == ["ocid1.nsg.oc1..test"]
    assert launch.shape == "VM.Standard.E5.Flex"
    assert launch.metadata["ssh_authorized_keys"] == "ssh-ed25519 AAA"
    assert oci.remaining(instance) is True
    oci.destroy(instance)
    assert compute.requests[-1] == instance
    assert oci.charges(instance) == 1.5
    bill = usage.requests[0]
    assert bill.query_type == "COST"
    assert bill.filter.tags[0].value == "tee-worker"


def b64url(raw) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def segment(value) -> str:
    return b64url(json.dumps(value, separators=(",", ":")).encode())


def attestation_http(key, measurement, snp_extra=None):
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )

    def handler(request):
        if request.url.path.endswith("/newSession"):
            nonce = request.url.params["nonce"]
            return httpx.Response(
                201,
                headers={"location": "session/test-session"},
                json={"status": "waiting", "nonce": nonce, "accept": []},
            )
        if request.url.path.endswith("/ratsd/chares"):
            return httpx.Response(
                200,
                json={
                    "cmw": "eyJldmlkZW5jZSI6InRlc3QifQ",
                    "eat_nonce": json.loads(request.content)["nonce"],
                    "eat_profile": "tag:github.com,2025:veraison/ratsd/cmw",
                },
            )
        nonce = json.loads(request.content)["eat_nonce"]
        snp = {
            "ear.status": "affirming",
            "ear.appraisal-policy-id": "policy:SEVSNP",
            "ear.trustworthiness-vector": {"hardware": 2, "runtime-opaque": 2},
            "ear.veraison.reference-values": {"swid:oci-sev-snp": {"value": measurement}},
        }
        if snp_extra:
            snp.update(snp_extra)
        claims = {
            "iat": int(time.time()),
            "eat_nonce": nonce,
            "submods": {"SEVSNP": snp},
        }
        header = segment({"alg": "ES256", "typ": "JWT"})
        payload = segment(claims)
        der = key.sign(f"{header}.{payload}".encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        token = f"{header}.{payload}.{b64url(signature)}"
        return httpx.Response(
            200,
            json={"status": "complete", "nonce": nonce, "result": token},
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return http, public_key


def test_oci_attestation_requires_fresh_affirming_snp_result() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    http, public_key = attestation_http(key, "measurement-1")
    oci = client(
        attestation_client=http,
        verifier_public_key_pem=public_key,
        attestation_policy_id="policy:SEVSNP",
    )
    claims = oci.attest(
        attester_url="https://attester.example:8895",
        verifier_url="https://verifier.example:8443",
        expected_measurements=["measurement-1"],
    )
    assert claims["submods"]["SEVSNP"]["ear.status"] == "affirming"


def test_oci_attestation_rejects_measurement_only_in_annotated_evidence() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    http, public_key = attestation_http(
        key,
        "measurement-other",
        snp_extra={"ear.veraison.annotated-evidence": {"note": "measurement-1 appeared here"}},
    )
    oci = client(attestation_client=http, verifier_public_key_pem=public_key)
    with pytest.raises(ProviderError, match="measurements"):
        oci.attest(
            attester_url="https://attester.example:8895",
            verifier_url="https://verifier.example:8443",
            expected_measurements=["measurement-1"],
        )


def test_oci_attestation_rejects_appraisal_without_reference_values() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    http, public_key = attestation_http(
        key,
        "measurement-1",
        snp_extra={"ear.veraison.reference-values": None},
    )
    oci = client(attestation_client=http, verifier_public_key_pem=public_key)
    with pytest.raises(ProviderError, match="reference values"):
        oci.attest(
            attester_url="https://attester.example:8895",
            verifier_url="https://verifier.example:8443",
            expected_measurements=["measurement-1"],
        )
