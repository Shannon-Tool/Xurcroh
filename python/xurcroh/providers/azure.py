from __future__ import annotations

import os
import time
from typing import Any

import httpx

from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import CloudRESTClient, allowed_source_ranges, instance_label


class AzureConfidentialClient:
    def __init__(
        self,
        *,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        credential: Any = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
        if not self.subscription_id:
            raise ProviderError("AZURE_SUBSCRIPTION_ID is required")
        self.resource_group = resource_group or os.environ.get(
            "AZURE_RESOURCE_GROUP", "xurcroh-tee"
        )
        if credential is None:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
        token = credential.get_token("https://management.azure.com/.default").token
        self._rest = CloudRESTClient(
            client
            or httpx.Client(
                base_url="https://management.azure.com",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
        )

    @property
    def _subscription(self) -> str:
        return f"/subscriptions/{self.subscription_id}"

    @property
    def _group(self) -> str:
        return f"{self._subscription}/resourceGroups/{self.resource_group}"

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name and "h100" not in gpu_name.lower():
            return []
        document = self._rest.request(
            "GET",
            f"{self._subscription}/providers/Microsoft.Compute/skus",
            params={"$filter": "name eq 'Standard_NCC40ads_H100_v5'"},
        )
        return [
            {
                "id": sku["name"],
                "provider_id": "azure",
                "tee_type": "sev-snp",
                "gpu_model": "H100 NVL",
                "gpu_memory_bytes": 94 * 1024**3,
                "cpu_vendor": "AMD",
                "country": sku["locations"][0],
                "attested": False,
                "attestation_required": True,
                "managed": True,
            }
            for sku in document.get("value", [])
            if sku.get("name") == "Standard_NCC40ads_H100_v5" and sku.get("locations")
        ]

    def _ensure_network(self, location: str, label: str, source_ranges: list[str]) -> str:
        self._rest.request(
            "PUT",
            self._group,
            params={"api-version": "2021-04-01"},
            json={"location": location},
        )
        network_root = f"{self._group}/providers/Microsoft.Network"
        nsg = f"{network_root}/networkSecurityGroups/{label}-nsg"
        security_rules = []
        for priority, name, protocol, port in (
            (100, "ssh", "Tcp", "22"),
            (110, "quic", "Udp", "9444"),
        ):
            security_rules.append(
                {
                    "name": name,
                    "properties": {
                        "protocol": protocol,
                        "sourcePortRange": "*",
                        "destinationPortRange": port,
                        "sourceAddressPrefixes": source_ranges,
                        "destinationAddressPrefix": "*",
                        "access": "Allow",
                        "priority": priority,
                        "direction": "Inbound",
                    },
                }
            )
        self._rest.request(
            "PUT",
            nsg,
            params={"api-version": "2023-11-01"},
            json={
                "location": location,
                "properties": {"securityRules": security_rules},
            },
        )
        vnet = f"{network_root}/virtualNetworks/{label}-vnet"
        self._rest.request(
            "PUT",
            vnet,
            params={"api-version": "2023-11-01"},
            json={
                "location": location,
                "properties": {
                    "addressSpace": {"addressPrefixes": ["10.42.0.0/16"]},
                    "subnets": [
                        {
                            "name": "tee-subnet",
                            "properties": {
                                "addressPrefix": "10.42.0.0/24",
                                "networkSecurityGroup": {"id": nsg},
                            },
                        }
                    ],
                },
            },
        )
        public_ip = f"{network_root}/publicIPAddresses/{label}-ip"
        self._rest.request(
            "PUT",
            public_ip,
            params={"api-version": "2023-11-01"},
            json={
                "location": location,
                "sku": {"name": "Standard"},
                "properties": {"publicIPAllocationMethod": "Static"},
            },
        )
        nic = f"{network_root}/networkInterfaces/{label}-nic"
        self._rest.request(
            "PUT",
            nic,
            params={"api-version": "2023-11-01"},
            json={
                "location": location,
                "properties": {
                    "ipConfigurations": [
                        {
                            "name": "primary",
                            "properties": {
                                "privateIPAllocationMethod": "Dynamic",
                                "subnet": {"id": f"{vnet}/subnets/tee-subnet"},
                                "publicIPAddress": {"id": public_ip},
                            },
                        }
                    ]
                },
            },
        )
        return nic

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        label = instance_label(str(values["label"]))
        public_key = values.get("ssh_public_key")
        if not isinstance(public_key, str) or not public_key.startswith(
            ("ssh-ed25519 ", "ssh-rsa ")
        ):
            raise ValueError("an SSH public key is required")
        location = str(offer["country"])
        source_ranges = allowed_source_ranges(values.get("allowed_source_ranges"))
        nic = self._ensure_network(location, label, source_ranges)
        image = values.get("image", "Ubuntu22.04")
        image_reference = (
            image
            if isinstance(image, dict)
            else {
                "publisher": "Canonical",
                "offer": "0001-com-ubuntu-server-jammy",
                "sku": "22_04-lts-gen2",
                "version": "latest",
            }
        )
        username = str(values.get("admin_username", "xurcroh"))
        payload = {
            "location": location,
            "properties": {
                "hardwareProfile": {"vmSize": "Standard_NCC40ads_H100_v5"},
                "storageProfile": {
                    "imageReference": image_reference,
                    "osDisk": {
                        "createOption": "FromImage",
                        "diskSizeGB": int(values.get("disk_gb", 64)),
                        "managedDisk": {"storageAccountType": "Premium_LRS"},
                    },
                },
                "osProfile": {
                    "computerName": label,
                    "adminUsername": username,
                    "linuxConfiguration": {
                        "disablePasswordAuthentication": True,
                        "ssh": {
                            "publicKeys": [
                                {
                                    "path": f"/home/{username}/.ssh/authorized_keys",
                                    "keyData": public_key,
                                }
                            ]
                        },
                    },
                },
                "securityProfile": {
                    "securityType": "ConfidentialVM",
                    "uefiSettings": {"secureBootEnabled": True, "vTpmEnabled": True},
                },
                "networkProfile": {"networkInterfaces": [{"id": nic}]},
            },
        }
        self._rest.request(
            "PUT",
            f"{self._group}/providers/Microsoft.Compute/virtualMachines/{label}",
            params={"api-version": "2024-07-01"},
            json=payload,
        )
        return label

    def destroy(self, instance_id: str) -> None:
        label = instance_label(str(instance_id).rsplit("/", 1)[-1])
        self._rest.request(
            "DELETE",
            f"{self._group}/providers/Microsoft.Compute/virtualMachines/{label}",
            params={"api-version": "2024-07-01"},
        )
        deadline = time.monotonic() + 300
        while self.remaining(label):
            if time.monotonic() >= deadline:
                raise ProviderError("Azure confidential instance deletion timed out")
            time.sleep(2)
        network_root = f"{self._group}/providers/Microsoft.Network"
        resources = (
            f"{network_root}/networkInterfaces/{label}-nic",
            f"{network_root}/publicIPAddresses/{label}-ip",
            f"{network_root}/virtualNetworks/{label}-vnet",
            f"{network_root}/networkSecurityGroups/{label}-nsg",
        )
        for path in resources:
            self._rest.request(
                "DELETE",
                path,
                params={"api-version": "2023-11-01"},
            )

    def remaining(self, instance_id: str) -> bool:
        document = self._rest.request(
            "GET",
            f"{self._group}/providers/Microsoft.Compute/virtualMachines/{instance_id}",
            params={"api-version": "2024-07-01"},
        )
        return document is not None

    def charges(self, instance_id: str) -> float:
        document = self._rest.request(
            "POST",
            f"{self._subscription}/providers/Microsoft.CostManagement/query",
            params={"api-version": "2023-11-01"},
            json={
                "type": "ActualCost",
                "dataSet": {
                    "granularity": "Daily",
                    "aggregation": {"totalCost": {"name": "PreTaxCost", "function": "Sum"}},
                    "filter": {
                        "dimensions": {
                            "name": "ResourceId",
                            "operator": "In",
                            "values": [instance_id],
                        }
                    },
                },
            },
        )
        rows = document.get("properties", {}).get("rows", [])
        return sum(float(row[0]) for row in rows)
