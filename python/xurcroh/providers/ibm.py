from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

from xurcroh.attestation import AttestationEvidence, DCAPCollateral, IntelDCAPVerifier
from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import instance_label


def _next_start(document: dict[str, Any]) -> str | None:
    next_page = document.get("next")
    if not isinstance(next_page, dict) or not next_page.get("href"):
        return None
    values = parse_qs(urlsplit(next_page["href"]).query).get("start")
    if not values or not values[0]:
        raise ProviderError("IBM pagination link has no start token")
    return values[0]


class IbmConfidentialClient:
    def __init__(
        self,
        *,
        account_id: str | None = None,
        region_id: str | None = None,
        vpc_client: Any = None,
        usage_client: Any = None,
    ) -> None:
        self.account_id = account_id or os.environ.get("IBM_ACCOUNT_ID")
        if not self.account_id:
            raise ProviderError("IBM_ACCOUNT_ID is required")
        self.region_id = region_id or os.environ.get("IBM_REGION", "us-south")
        if vpc_client is None:
            from ibm_vpc import VpcV1

            vpc_client = VpcV1.new_instance()
            vpc_client.set_service_url(f"https://{self.region_id}.iaas.cloud.ibm.com")
        if usage_client is None:
            from ibm_platform_services import UsageReportsV4

            usage_client = UsageReportsV4.new_instance()
        self._vpc = vpc_client
        self._usage = usage_client

    @staticmethod
    def _call(operation: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            raise ProviderError(f"IBM Cloud {operation} failed") from error

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name:
            return []
        offers = []
        start = None
        for _ in range(1000):
            response = self._call(
                "profile discovery",
                self._vpc.list_instance_profiles,
                limit=100,
                start=start,
            ).get_result()
            for profile in response.get("profiles", []):
                modes = profile.get("confidential_compute_modes", {})
                gpu_count = profile.get("gpu_count", {}).get("value", 0)
                if (
                    profile.get("status") != "current"
                    or "tdx" not in modes.get("values", [])
                    or gpu_count
                ):
                    continue
                for zone in profile.get("zones", []):
                    offers.append(
                        {
                            "id": profile["name"],
                            "provider_id": "ibm",
                            "tee_type": "tdx",
                            "gpu_model": None,
                            "gpu_memory_bytes": 0,
                            "cpu_vendor": "Intel",
                            "country": zone["name"],
                            "attested": False,
                            "attestation_required": True,
                            "managed": True,
                        }
                    )
            start = _next_start(response)
            if start is None:
                return offers
        raise ProviderError("IBM profile pagination did not terminate")

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        from ibm_vpc import vpc_v1

        label = instance_label(str(values["label"]))
        security_groups = values.get("security_group_ids")
        ssh_keys = values.get("ssh_key_ids")
        if not isinstance(security_groups, list) or not security_groups:
            raise ValueError("scoped IBM security groups are required")
        if not isinstance(ssh_keys, list) or not ssh_keys:
            raise ValueError("at least one IBM SSH key ID is required")
        prototype = (
            vpc_v1.InstancePrototypeInstanceByImageInstanceByImageInstanceByNetworkInterface(
                name=label,
                image=vpc_v1.ImageIdentityById(id=str(values["image_id"])),
                zone=vpc_v1.ZoneIdentityByName(name=str(values.get("zone", offer["country"]))),
                profile=vpc_v1.InstanceProfileIdentityByName(name=str(offer["id"])),
                vpc=vpc_v1.VPCIdentityById(id=str(values["vpc_id"])),
                primary_network_interface=vpc_v1.NetworkInterfacePrototype(
                    name="primary",
                    subnet=vpc_v1.SubnetIdentityById(id=str(values["subnet_id"])),
                    security_groups=[
                        vpc_v1.SecurityGroupIdentityById(id=group_id)
                        for group_id in security_groups
                    ],
                    allow_ip_spoofing=False,
                ),
                keys=[vpc_v1.KeyIdentityById(id=key_id) for key_id in ssh_keys],
                confidential_compute_mode="tdx",
                enable_secure_boot=True,
                metadata_service=vpc_v1.InstanceMetadataServicePrototype(
                    enabled=True,
                    protocol="https",
                    response_hop_limit=2,
                ),
            )
        )
        result = self._call("instance creation", self._vpc.create_instance, prototype).get_result()
        instance_id = result.get("id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ProviderError("IBM Cloud did not return an instance ID")
        return instance_id

    def destroy(self, instance_id: str) -> None:
        self._call("instance deletion", self._vpc.delete_instance, instance_id)

    def remaining(self, instance_id: str) -> bool:
        try:
            result = self._vpc.get_instance(instance_id).get_result()
        except Exception as error:
            if getattr(error, "code", None) == 404:
                return False
            raise ProviderError("IBM Cloud instance lookup failed") from error
        return result.get("status") not in {"deleting", "deleted", "failed"}

    def charges(self, instance_id: str) -> float:
        instance = self._call("instance lookup", self._vpc.get_instance, instance_id).get_result()
        resource_instance_id = instance.get("resource_instance_id") or instance.get("crn")
        if not resource_instance_id:
            raise ProviderError("IBM instance has no billable resource identity")
        total = 0.0
        start = None
        for _ in range(1000):
            document = self._call(
                "usage query",
                self._usage.get_resource_usage_account,
                account_id=self.account_id,
                billingmonth=datetime.now(UTC).strftime("%Y-%m"),
                resource_instance_id=resource_instance_id,
                names=False,
                limit=100,
                start=start,
            ).get_result()
            for resource in document.get("resources", []):
                if resource.get("resource_instance_id") != resource_instance_id:
                    continue
                if resource.get("currency_code") != "USD":
                    raise ProviderError("non-USD IBM bills require explicit FX conversion")
                if resource.get("billable") is False:
                    continue
                for metric in resource.get("usage", []):
                    if not metric.get("non_chargeable", False):
                        total += float(metric.get("cost", 0.0))
            start = _next_start(document)
            if start is None:
                return total
        raise ProviderError("IBM usage pagination did not terminate")

    @staticmethod
    def verify_attestation(
        *,
        evidence: AttestationEvidence,
        quote: bytes,
        collateral: DCAPCollateral,
        expected_nonce: bytes,
    ) -> Any:
        verifier = IntelDCAPVerifier(expected_nonce=expected_nonce)
        return verifier.verify(evidence, quote=quote, collateral=collateral)
