from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import CloudRESTClient, allowed_source_ranges, instance_label

ATTESTATION_ISSUER = "https://attest.cn-beijing.aliyuncs.com"


def _decode_segment(value: str) -> bytes:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ProviderError("attestation token contains invalid base64url")
    try:
        return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as error:
        raise ProviderError("attestation token contains invalid base64url") from error


def _json_segment(value: str) -> dict[str, Any]:
    try:
        document = json.loads(_decode_segment(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderError("attestation token contains invalid JSON") from error
    if not isinstance(document, dict):
        raise ProviderError("attestation token segments must be objects")
    return document


class AlibabaConfidentialClient:
    def __init__(
        self,
        *,
        region_id: str | None = None,
        ecs_client: Any = None,
        billing_client: Any = None,
        attestation_client: httpx.Client | None = None,
        attestation_policy_ids: list[str] | None = None,
    ) -> None:
        self.region_id = region_id or os.environ.get("ALIBABA_REGION_ID", "cn-beijing")
        if ecs_client is None or billing_client is None:
            from alibabacloud_bssopenapi20171214.client import Client as BillingClient
            from alibabacloud_credentials.client import Client as CredentialClient
            from alibabacloud_ecs20140526.client import Client as ECSClient
            from alibabacloud_tea_openapi import models as open_api_models

            credential = CredentialClient()
            ecs_config = open_api_models.Config(
                credential=credential,
                region_id=self.region_id,
                endpoint=f"ecs.{self.region_id}.aliyuncs.com",
                read_timeout=60_000,
                connect_timeout=10_000,
                tls_min_version="TLSv1.2",
            )
            billing_config = open_api_models.Config(
                credential=credential,
                region_id=self.region_id,
                endpoint="business.aliyuncs.com",
                read_timeout=60_000,
                connect_timeout=10_000,
                tls_min_version="TLSv1.2",
            )
            ecs_client = ecs_client or ECSClient(ecs_config)
            billing_client = billing_client or BillingClient(billing_config)
        self._ecs = ecs_client
        self._billing = billing_client
        self._attestation_http = attestation_client or httpx.Client(
            base_url=ATTESTATION_ISSUER, timeout=60
        )
        self._attestation_rest = CloudRESTClient(self._attestation_http)
        configured_policies = attestation_policy_ids
        if configured_policies is None:
            configured_policies = os.environ.get("ALIBABA_ATTESTATION_POLICY_IDS", "").split(",")
        self._attestation_policy_ids = tuple(
            policy.strip() for policy in configured_policies if policy.strip()
        )

    @staticmethod
    def _call(operation: str, function: Any, request: Any) -> Any:
        try:
            return function(request)
        except Exception as error:
            raise ProviderError(f"Alibaba Cloud {operation} failed") from error

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name and "h100" not in gpu_name.lower():
            return []
        from alibabacloud_ecs20140526 import models

        request = models.DescribeAvailableResourceRequest(
            region_id=self.region_id,
            destination_resource="InstanceType",
            resource_type="instance",
            spot_strategy="SpotAsPriceGo",
        )
        result = self._call(
            "available-resource discovery", self._ecs.describe_available_resource, request
        )
        offers = []
        zones = result.body.available_zones
        for zone in zones.available_zone if zones else []:
            if zone.status != "Available":
                continue
            resources = zone.available_resources
            for available in resources.available_resource if resources else []:
                supported = available.supported_resources
                for resource in supported.supported_resource if supported else []:
                    instance_type = resource.value
                    if not instance_type.startswith("ecs.gn8v-tee."):
                        continue
                    if resource.status not in {"Available", "WithStock"}:
                        continue
                    offers.append(
                        {
                            "id": instance_type,
                            "provider_id": "alibaba",
                            "tee_type": "tdx",
                            "gpu_model": "H100",
                            "gpu_memory_bytes": 96 * 1024**3,
                            "cpu_vendor": "Intel",
                            "country": zone.zone_id,
                            "attested": False,
                            "attestation_required": True,
                            "managed": True,
                        }
                    )
        return offers

    def _validate_image(self, image_id: str, instance_type: str) -> None:
        from alibabacloud_ecs20140526 import models

        request = models.DescribeImagesRequest(
            region_id=self.region_id,
            image_id=image_id,
            instance_type=instance_type,
            status="Available",
        )
        result = self._call("image validation", self._ecs.describe_images, request)
        images = result.body.images
        matches = images.image if images else []
        if len(matches) != 1:
            raise ProviderError("the requested Alibaba Cloud image is unavailable")
        image = matches[0]
        approved = (
            image.image_id == image_id
            and image.status == "Available"
            and image.image_owner_alias == "system"
            and image.architecture == "x86_64"
            and image.ostype == "linux"
            and "Alibaba Cloud Linux 3" in image.image_name
        )
        if not approved:
            raise ProviderError("gn8v-tee requires an approved x86_64 Alibaba Cloud Linux 3 image")

    def _create_security_group(self, label: str, vpc_id: str, source_ranges: list[str]) -> str:
        from alibabacloud_ecs20140526 import models

        create_request = models.CreateSecurityGroupRequest(
            region_id=self.region_id,
            security_group_name=f"xurcroh-{label}",
            description="Xurcroh confidential worker ingress",
            security_group_type="normal",
            vpc_id=vpc_id,
        )
        created = self._call(
            "security-group creation", self._ecs.create_security_group, create_request
        )
        security_group_id = created.body.security_group_id
        for source_range in source_ranges:
            for protocol, port in (("tcp", "22/22"), ("udp", "9444/9444")):
                request = models.AuthorizeSecurityGroupRequest(
                    region_id=self.region_id,
                    security_group_id=security_group_id,
                    ip_protocol=protocol,
                    port_range=port,
                    source_cidr_ip=source_range,
                    policy="accept",
                    priority="1",
                    nic_type="intranet",
                )
                self._call(
                    "security-group authorization",
                    self._ecs.authorize_security_group,
                    request,
                )
        if not isinstance(security_group_id, str) or not security_group_id:
            raise ProviderError("Alibaba Cloud did not return a security group ID")
        return security_group_id

    @staticmethod
    def _required(values: dict[str, Any], name: str) -> str:
        value = values.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} is required")
        return value

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        from alibabacloud_ecs20140526 import models

        label = instance_label(str(values["label"]))
        image_id = self._required(values, "image_id")
        key_pair_name = self._required(values, "key_pair_name")
        vpc_id = self._required(values, "vpc_id")
        v_switch_id = self._required(values, "v_switch_id")
        instance_type = str(offer["id"])
        if not instance_type.startswith("ecs.gn8v-tee."):
            raise ValueError("Alibaba confidential GPU offers must use gn8v-tee")
        self._validate_image(image_id, instance_type)
        source_ranges = allowed_source_ranges(values.get("allowed_source_ranges"))
        security_group_id = self._create_security_group(label, vpc_id, source_ranges)
        request = models.RunInstancesRequest(
            region_id=self.region_id,
            zone_id=str(offer["country"]),
            image_id=image_id,
            instance_type=instance_type,
            instance_name=label,
            host_name=label,
            amount=1,
            min_amount=1,
            key_pair_name=key_pair_name,
            security_group_id=security_group_id,
            v_switch_id=v_switch_id,
            internet_charge_type="PayByTraffic",
            internet_max_bandwidth_out=int(values.get("bandwidth_mbps", 100)),
            http_tokens="required",
            http_endpoint="enabled",
            http_put_response_hop_limit=2,
            deletion_protection=False,
            security_options=models.RunInstancesRequestSecurityOptions(
                confidential_computing_mode="TDX",
                enable_secure_boot=True,
            ),
            system_disk=models.RunInstancesRequestSystemDisk(
                category="cloud_essd",
                size=str(int(values.get("disk_gb", 64))),
                encrypted="true",
            ),
            tag=[
                models.RunInstancesRequestTag(key="xurcroh:managed", value="true"),
                models.RunInstancesRequestTag(key="xurcroh:instance", value=label),
            ],
        )
        result = self._call("instance creation", self._ecs.run_instances, request)
        instance_ids = result.body.instance_id_sets.instance_id_set
        if len(instance_ids) != 1:
            raise ProviderError("Alibaba Cloud did not return exactly one instance ID")
        return str(instance_ids[0])

    def destroy(self, instance_id: str) -> None:
        from alibabacloud_ecs20140526 import models

        lookup_request = models.DescribeInstancesRequest(
            region_id=self.region_id,
            instance_ids=json.dumps([instance_id], separators=(",", ":")),
        )
        lookup = self._call("instance lookup", self._ecs.describe_instances, lookup_request)
        instances = lookup.body.instances.instance if lookup.body.instances else []
        if len(instances) > 1:
            raise ProviderError("Alibaba Cloud returned duplicate instance identities")
        security_groups = []
        if instances:
            group_ids = instances[0].security_group_ids
            security_groups = list(group_ids.security_group_id or []) if group_ids else []
        request = models.DeleteInstanceRequest(
            instance_id=instance_id,
            force=True,
            terminate_subscription=True,
        )
        self._call("instance deletion", self._ecs.delete_instance, request)
        deadline = time.monotonic() + 300
        while self.remaining(instance_id):
            if time.monotonic() >= deadline:
                raise ProviderError("Alibaba Cloud instance deletion timed out")
            time.sleep(2)
        for security_group_id in security_groups:
            delete_group = models.DeleteSecurityGroupRequest(
                region_id=self.region_id,
                security_group_id=security_group_id,
            )
            self._call("security-group deletion", self._ecs.delete_security_group, delete_group)

    def remaining(self, instance_id: str) -> bool:
        from alibabacloud_ecs20140526 import models

        request = models.DescribeInstancesRequest(
            region_id=self.region_id,
            instance_ids=json.dumps([instance_id], separators=(",", ":")),
        )
        result = self._call("instance lookup", self._ecs.describe_instances, request)
        instances = result.body.instances
        return bool(instances and instances.instance)

    def charges(self, instance_id: str) -> float:
        from alibabacloud_bssopenapi20171214 import models

        total = 0.0
        next_token = None
        for _ in range(1000):
            request = models.DescribeInstanceBillRequest(
                billing_cycle=datetime.now(UTC).strftime("%Y-%m"),
                instance_id=instance_id,
                granularity="DAILY",
                is_billing_item=False,
                max_results=300,
                next_token=next_token,
            )
            result = self._call("billing query", self._billing.describe_instance_bill, request)
            if result.body.success is False:
                raise ProviderError("Alibaba Cloud billing query was unsuccessful")
            data = result.body.data
            for item in data.items if data and data.items else []:
                if item.currency not in {None, "USD"}:
                    raise ProviderError("non-USD Alibaba bills require explicit FX conversion")
                total += float(item.pretax_amount or 0.0)
            next_token = data.next_token if data else None
            if not next_token:
                return total
        raise ProviderError("Alibaba Cloud billing pagination did not terminate")

    def attest(
        self,
        evidence: str,
        *,
        runtime_data: dict[str, str],
        policy_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        _decode_segment(evidence)
        policies = tuple(policy_ids or self._attestation_policy_ids)
        if not policies:
            raise ProviderError("at least one Alibaba attestation policy is required")
        if set(runtime_data) != {"raw"} or not runtime_data["raw"]:
            raise ProviderError("attestation runtime_data must bind a raw challenge")
        try:
            response = self._attestation_http.post(
                "/v1/attestation",
                json={
                    "tee": "acstdxnvgpu",
                    "evidence": evidence,
                    "policy_ids": list(policies),
                    "runtime_data": runtime_data,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderError("Alibaba attestation request failed") from error
        token = response.text.strip()
        if len(token) > 131_072:
            raise ProviderError("Alibaba attestation token is too large")
        parts = token.split(".")
        if len(parts) != 3:
            raise ProviderError("Alibaba attestation did not return a compact JWS")
        header = _json_segment(parts[0])
        claims = _json_segment(parts[1])
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise ProviderError("Alibaba attestation token has an invalid header")
        jwks = self._attestation_rest.request("GET", "/jwks.json")
        keys = [
            key
            for key in jwks.get("keys", [])
            if key.get("kid") == header["kid"]
            and key.get("kty") == "RSA"
            and key.get("alg") == "RS256"
        ]
        if len(keys) != 1:
            raise ProviderError("Alibaba attestation signing key is unavailable")
        key = keys[0]
        try:
            modulus = int.from_bytes(_decode_segment(key["n"]), "big")
            exponent = int.from_bytes(_decode_segment(key["e"]), "big")
            public_key = rsa.RSAPublicNumbers(exponent, modulus).public_key()
            public_key.verify(
                _decode_segment(parts[2]),
                f"{parts[0]}.{parts[1]}".encode("ascii"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except (InvalidSignature, KeyError, TypeError, ValueError) as error:
            raise ProviderError("Alibaba attestation signature is invalid") from error
        self._validate_claims(claims, policies, runtime_data)
        return claims

    @staticmethod
    def _validate_claims(
        claims: dict[str, Any], policies: tuple[str, ...], runtime_data: dict[str, str]
    ) -> None:
        audience = claims.get("aud")
        valid_audience = audience == ATTESTATION_ISSUER or (
            isinstance(audience, list) and ATTESTATION_ISSUER in audience
        )
        if claims.get("iss") != ATTESTATION_ISSUER or not valid_audience:
            raise ProviderError("Alibaba attestation issuer or audience is invalid")
        now = time.time()
        timestamps = [claims.get(name) for name in ("iat", "nbf", "exp")]
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) for value in timestamps
        ):
            raise ProviderError("Alibaba attestation timestamps are missing")
        if claims["nbf"] > now + 60 or claims["iat"] > now + 60:
            raise ProviderError("Alibaba attestation token is not yet valid")
        if claims["exp"] <= now - 60 or claims["exp"] <= claims["iat"]:
            raise ProviderError("Alibaba attestation token is expired")
        if claims.get("tee") != "acstdxnvgpu":
            raise ProviderError("Alibaba attestation is not composite CPU/GPU evidence")
        customized = claims.get("customized_claims")
        if not isinstance(customized, dict) or customized.get("runtime_data") != runtime_data:
            raise ProviderError("Alibaba attestation runtime binding is invalid")
        reports = claims.get("evaluation-reports")
        if not isinstance(reports, list):
            raise ProviderError("Alibaba attestation policy reports are missing")
        evaluated = {report.get("policy-id") for report in reports if isinstance(report, dict)}
        if not set(policies).issubset(evaluated):
            raise ProviderError("Alibaba attestation policy appraisal is incomplete")
