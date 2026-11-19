from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from xurcroh.errors import ProviderError
from xurcroh.providers.alibaba import AlibabaConfidentialClient


def response(body):
    return SimpleNamespace(body=body)


class FakeECS:
    def __init__(self) -> None:
        self.requests = []

    def describe_available_resource(self, request):
        self.requests.append(request)
        resource = SimpleNamespace(value="ecs.gn8v-tee.4xlarge", status="Available")
        supported = SimpleNamespace(supported_resource=[resource])
        available = SimpleNamespace(type="InstanceType", supported_resources=supported)
        zone = SimpleNamespace(
            zone_id="cn-beijing-l",
            status="Available",
            available_resources=SimpleNamespace(available_resource=[available]),
        )
        return response(SimpleNamespace(available_zones=SimpleNamespace(available_zone=[zone])))

    def describe_images(self, request):
        self.requests.append(request)
        image = SimpleNamespace(
            image_id=request.image_id,
            image_name="Alibaba Cloud Linux 3.2104 LTS 64-bit",
            architecture="x86_64",
            ostype="linux",
            status="Available",
            image_owner_alias="system",
        )
        return response(SimpleNamespace(images=SimpleNamespace(image=[image])))

    def create_security_group(self, request):
        self.requests.append(request)
        return response(SimpleNamespace(security_group_id="sg-test"))

    def authorize_security_group(self, request):
        self.requests.append(request)
        return response(SimpleNamespace())

    def run_instances(self, request):
        self.requests.append(request)
        ids = SimpleNamespace(instance_id_set=["i-test"])
        return response(SimpleNamespace(instance_id_sets=ids))

    def describe_instances(self, request):
        self.requests.append(request)
        return response(SimpleNamespace(instances=SimpleNamespace(instance=[SimpleNamespace()])))

    def delete_instance(self, request):
        self.requests.append(request)
        return response(SimpleNamespace())


class FakeBilling:
    def __init__(self) -> None:
        self.requests = []

    def describe_instance_bill(self, request):
        self.requests.append(request)
        data = SimpleNamespace(
            items=[
                SimpleNamespace(pretax_amount=1.25, currency="USD"),
                SimpleNamespace(pretax_amount=1.0, currency="USD"),
            ],
            next_token=None,
        )
        return response(SimpleNamespace(success=True, data=data))


def client(ecs=None, billing=None, attestation_client=None, attestation_policy_ids=None):
    return AlibabaConfidentialClient(
        region_id="cn-beijing",
        ecs_client=ecs or FakeECS(),
        billing_client=billing or FakeBilling(),
        attestation_client=attestation_client,
        attestation_policy_ids=attestation_policy_ids,
    )


def test_alibaba_confidential_gpu_lifecycle() -> None:
    ecs = FakeECS()
    billing = FakeBilling()
    alibaba = client(ecs, billing)
    offer = alibaba.offers()[0]
    assert offer["id"] == "ecs.gn8v-tee.4xlarge"
    assert offer["tee_type"] == "tdx"
    assert offer["gpu_memory_bytes"] == 96 * 1024**3
    instance = alibaba.create(
        offer,
        label="tee-worker",
        image_id="aliyun_3_x64_20G_alibase_20250117.vhd",
        key_pair_name="tee-key",
        vpc_id="vpc-test",
        v_switch_id="vsw-test",
        disk_gb=64,
        allowed_source_ranges=["192.0.2.10/32"],
    )
    assert instance == "i-test"
    run = next(request for request in ecs.requests if hasattr(request, "security_options"))
    assert run.security_options.confidential_computing_mode == "TDX"
    assert run.security_options.enable_secure_boot is True
    assert run.system_disk.encrypted == "true"
    assert run.http_tokens == "required"
    assert run.key_pair_name == "tee-key"
    assert run.security_group_id == "sg-test"
    ingress = [
        request
        for request in ecs.requests
        if hasattr(request, "source_cidr_ip") and request.source_cidr_ip
    ]
    assert len(ingress) == 2
    assert {request.port_range for request in ingress} == {"22/22", "9444/9444"}
    assert alibaba.remaining(instance) is True
    alibaba.destroy(instance)
    assert ecs.requests[-1].instance_id == instance
    assert alibaba.charges(instance) == 2.25
    assert billing.requests[0].instance_id == instance


def segment(value) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def signed_token(key, claims, kid="key-1") -> str:
    header = segment({"alg": "RS256", "kid": kid, "typ": "JWT"})
    payload = segment(claims)
    signature = key.sign(f"{header}.{payload}".encode(), padding.PKCS1v15(), hashes.SHA256())
    encoded = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{header}.{payload}.{encoded}"


def attestation_fixture(claims):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = key.public_key().public_numbers()

    def encode_int(value):
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    token = signed_token(key, claims)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "keys": [
                        {
                            "kty": "RSA",
                            "kid": "key-1",
                            "use": "sig",
                            "alg": "RS256",
                            "n": encode_int(numbers.n),
                            "e": encode_int(numbers.e),
                        }
                    ]
                },
            )
        return httpx.Response(200, text=token)

    http = httpx.Client(
        base_url="https://attest.cn-beijing.aliyuncs.com",
        transport=httpx.MockTransport(handler),
    )
    return http


def test_alibaba_attestation_verifies_oidc_and_policy_binding() -> None:
    now = int(time.time())
    claims = {
        "iss": "https://attest.cn-beijing.aliyuncs.com",
        "aud": "https://attest.cn-beijing.aliyuncs.com",
        "iat": now,
        "nbf": now,
        "exp": now + 600,
        "tee": "acstdxnvgpu",
        "evaluation-reports": [{"policy-id": "policy-gpu"}],
        "customized_claims": {"runtime_data": {"raw": "Y2hhbGxlbmdl"}},
    }
    alibaba = client(
        attestation_client=attestation_fixture(claims),
        attestation_policy_ids=["policy-gpu"],
    )
    result = alibaba.attest("ZXZpZGVuY2U", runtime_data={"raw": "Y2hhbGxlbmdl"})
    assert result["tee"] == "acstdxnvgpu"


def test_alibaba_attestation_rejects_missing_policy() -> None:
    now = int(time.time())
    claims = {
        "iss": "https://attest.cn-beijing.aliyuncs.com",
        "aud": "https://attest.cn-beijing.aliyuncs.com",
        "iat": now,
        "nbf": now,
        "exp": now + 600,
        "tee": "acstdxnvgpu",
        "evaluation-reports": [],
        "customized_claims": {"runtime_data": {"raw": "Y2hhbGxlbmdl"}},
    }
    alibaba = client(
        attestation_client=attestation_fixture(claims),
        attestation_policy_ids=["policy-gpu"],
    )
    with pytest.raises(ProviderError, match="policy"):
        alibaba.attest("ZXZpZGVuY2U", runtime_data={"raw": "Y2hhbGxlbmdl"})


def test_alibaba_rejects_unapproved_image() -> None:
    ecs = FakeECS()
    image = SimpleNamespace(
        image_id="ubuntu",
        image_name="Ubuntu 22.04",
        architecture="x86_64",
        ostype="linux",
        status="Available",
        image_owner_alias="system",
    )
    ecs.describe_images = lambda request: response(
        SimpleNamespace(images=SimpleNamespace(image=[image]))
    )
    alibaba = client(ecs)
    with pytest.raises(ProviderError, match="Alibaba Cloud Linux 3"):
        alibaba.create(
            {"country": "cn-beijing-l", "id": "ecs.gn8v-tee.4xlarge"},
            label="tee-worker",
            image_id="ubuntu",
            key_pair_name="tee-key",
            vpc_id="vpc-test",
            v_switch_id="vsw-test",
            allowed_source_ranges=["192.0.2.10/32"],
        )
