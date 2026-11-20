from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

import httpx

from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import CloudRESTClient, allowed_source_ranges, instance_label


def _network_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _instance_network_tag(label: str) -> str:
    return f"xurcroh-{_network_digest(label)[:24]}"


def _firewall_name(label: str) -> str:
    return f"xurcroh-fw-{_network_digest(label)[:20]}"


class GoogleConfidentialClient:
    def __init__(
        self,
        *,
        project_id: str | None = None,
        billing_export_table: str | None = None,
        credential: Any = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ProviderError("GOOGLE_CLOUD_PROJECT is required")
        self.billing_export_table = billing_export_table or os.environ.get(
            "GOOGLE_BILLING_EXPORT_TABLE"
        )
        if (
            self.billing_export_table is not None
            and re.fullmatch(
                r"[A-Za-z0-9-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_-]+",
                self.billing_export_table,
            )
            is None
        ):
            raise ValueError("billing export must be a fully qualified BigQuery table")
        if credential is None:
            from google.auth import default

            credential, _ = default()
        if hasattr(credential, "get_token"):
            token = credential.get_token("https://www.googleapis.com/auth/cloud-platform").token
        else:
            from google.auth.transport.requests import Request

            credential.refresh(Request())
            token = credential.token
        self._rest = CloudRESTClient(
            client
            or httpx.Client(
                base_url="https://compute.googleapis.com/compute/v1",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
        )

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name and "h100" not in gpu_name.lower():
            return []
        document = self._rest.request("GET", f"projects/{self.project_id}/zones")
        offers = []
        for zone in document.get("items", []):
            if zone.get("status") != "UP":
                continue
            name = zone["name"]
            machine = self._rest.request(
                "GET",
                f"projects/{self.project_id}/zones/{name}/machineTypes/a3-highgpu-1g",
            )
            if machine is None:
                continue
            offers.append(
                {
                    "id": "a3-highgpu-1g",
                    "provider_id": "gcp",
                    "tee_type": "tdx",
                    "gpu_model": "H100",
                    "gpu_memory_bytes": 80 * 1024**3,
                    "cpu_vendor": "Intel",
                    "country": name,
                    "attested": False,
                    "attestation_required": True,
                    "managed": True,
                }
            )
        return offers

    def _ensure_firewall(self, label: str, source_ranges: list[str]) -> None:
        name = _firewall_name(label)
        path = f"projects/{self.project_id}/global/firewalls/{name}"
        payload = {
            "name": name,
            "network": "global/networks/default",
            "direction": "INGRESS",
            "priority": 1000,
            "sourceRanges": source_ranges,
            "allowed": [
                {"IPProtocol": "tcp", "ports": ["22"]},
                {"IPProtocol": "udp", "ports": ["9444"]},
            ],
            "targetTags": [_instance_network_tag(label)],
        }
        if self._rest.request("GET", path) is None:
            self._rest.request(
                "POST",
                f"projects/{self.project_id}/global/firewalls",
                json=payload,
            )
        else:
            self._rest.request("PATCH", path, json=payload)

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        label = instance_label(str(values["label"])).lower()
        if re.fullmatch(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?", label) is None:
            raise ValueError("GCP instance names must be RFC 1035 labels")
        public_key = values.get("ssh_public_key")
        if not isinstance(public_key, str) or not public_key.startswith(
            ("ssh-ed25519 ", "ssh-rsa ")
        ):
            raise ValueError("an SSH public key is required")
        source_ranges = allowed_source_ranges(values.get("allowed_source_ranges"))
        self._ensure_firewall(label, source_ranges)
        zone = str(offer["country"])
        source_image = values.get(
            "source_image",
            "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts",
        )
        payload = {
            "name": label,
            "labels": {"xurcroh_instance": label.replace("-", "_")},
            "tags": {"items": [_instance_network_tag(label)]},
            "machineType": f"zones/{zone}/machineTypes/a3-highgpu-1g",
            "confidentialInstanceConfig": {"confidentialInstanceType": "TDX"},
            "scheduling": {
                "onHostMaintenance": "TERMINATE",
                "automaticRestart": True,
                "provisioningModel": "SPOT",
            },
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "initializeParams": {
                        "sourceImage": source_image,
                        "diskSizeGb": str(int(values.get("disk_gb", 64))),
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "name": "nic0",
                    "network": "global/networks/default",
                    "accessConfigs": [
                        {
                            "kind": "compute#accessConfig",
                            "name": "External NAT",
                            "type": "ONE_TO_ONE_NAT",
                            "networkTier": "PREMIUM",
                        }
                    ],
                }
            ],
            "shieldedInstanceConfig": {
                "enableSecureBoot": True,
                "enableVtpm": True,
                "enableIntegrityMonitoring": True,
            },
            "metadata": {"items": [{"key": "ssh-keys", "value": f"xurcroh:{public_key}"}]},
        }
        self._rest.request(
            "POST",
            f"projects/{self.project_id}/zones/{zone}/instances",
            json=payload,
        )
        return f"zones/{zone}/instances/{label}"

    def destroy(self, instance_id: str) -> None:
        zone = instance_id.split("/")[-3]
        label = instance_id.rsplit("/", 1)[-1]
        firewall = f"projects/{self.project_id}/global/firewalls/{_firewall_name(label)}"
        try:
            self._rest.request(
                "DELETE",
                f"projects/{self.project_id}/zones/{zone}/instances/{label}",
            )
            deadline = time.monotonic() + 300
            while self.remaining(instance_id):
                if time.monotonic() >= deadline:
                    raise ProviderError("Google Cloud confidential instance deletion timed out")
                time.sleep(2)
        finally:
            self._rest.request("DELETE", firewall)
            deadline = time.monotonic() + 300
            while self._rest.request("GET", firewall) is not None:
                if time.monotonic() >= deadline:
                    raise ProviderError(
                        f"Google Cloud firewall rule {_firewall_name(label)} "
                        "could not be confirmed deleted"
                    )
                time.sleep(2)

    def remaining(self, instance_id: str) -> bool:
        zone = instance_id.split("/")[-3]
        label = instance_id.rsplit("/", 1)[-1]
        document = self._rest.request(
            "GET",
            f"projects/{self.project_id}/zones/{zone}/instances/{label}",
        )
        return document is not None

    def charges(self, instance_id: str) -> float:
        if self.billing_export_table is None:
            raise ProviderError("GOOGLE_BILLING_EXPORT_TABLE is required for charge reporting")
        label = instance_id.rsplit("/", 1)[-1].replace("-", "_")
        document = self._rest.request(
            "POST",
            f"https://bigquery.googleapis.com/bigquery/v2/projects/{self.project_id}/queries",
            json={
                # The table identifier is strictly validated; instance values are bound.
                "query": (
                    f"SELECT COALESCE(SUM(cost), 0) AS total_cost "  # noqa: S608
                    f"FROM `{self.billing_export_table}` "
                    "WHERE EXISTS ("
                    "SELECT 1 FROM UNNEST(labels) AS label "
                    "WHERE label.key = 'xurcroh_instance' "
                    "AND label.value = @instance_label)"
                ),
                "useLegacySql": False,
                "parameterMode": "NAMED",
                "queryParameters": [
                    {
                        "name": "instance_label",
                        "parameterType": {"type": "STRING"},
                        "parameterValue": {"value": label},
                    }
                ],
            },
        )
        rows = document.get("rows", [])
        if not rows:
            return 0.0
        return float(rows[0]["f"][0]["v"])
