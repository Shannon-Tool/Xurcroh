from __future__ import annotations

import json

import httpx

from xurcroh.providers.azure import AzureConfidentialClient
from xurcroh.providers.gcp import GoogleConfidentialClient


class Credential:
    def get_token(self, _scope: str):
        return type("Token", (), {"token": "test-token"})()


def test_azure_confidential_lifecycle() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/skus"):
            return httpx.Response(
                200,
                json={
                    "value": [
                        {
                            "name": "Standard_NCC40ads_H100_v5",
                            "locations": ["eastus"],
                        }
                    ]
                },
            )
        if path.endswith("/Microsoft.CostManagement/query"):
            return httpx.Response(200, json={"properties": {"rows": [[2.5, "USD"]]}})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.method == "GET" and "virtualMachines/tee-worker" in path:
            return httpx.Response(404, json={})
        return httpx.Response(200, json={"properties": {"provisioningState": "Succeeded"}})

    client = AzureConfidentialClient(
        subscription_id="sub",
        credential=Credential(),
        client=httpx.Client(
            base_url="https://management.azure.com",
            transport=httpx.MockTransport(handler),
        ),
    )
    offer = client.offers()[0]
    assert offer["tee_type"] == "sev-snp"
    assert offer["gpu_memory_bytes"] == 94 * 1024**3
    instance = client.create(
        offer,
        image="Ubuntu22.04",
        disk_gb=64,
        label="tee-worker",
        ssh_public_key="ssh-ed25519 AAA",
        allowed_source_ranges=["192.0.2.10/32"],
    )
    assert instance == "tee-worker"
    payload = next(
        json.loads(request.content)
        for request in seen
        if request.method == "PUT" and "virtualMachines" in str(request.url)
    )
    assert payload["properties"]["securityProfile"]["securityType"] == "ConfidentialVM"
    assert payload["properties"]["hardwareProfile"]["vmSize"] == "Standard_NCC40ads_H100_v5"
    nsg = next(
        json.loads(request.content)
        for request in seen
        if request.method == "PUT" and "networkSecurityGroups" in str(request.url)
    )
    rules = nsg["properties"]["securityRules"]
    assert {tuple(rule["properties"]["sourceAddressPrefixes"]) for rule in rules} == {
        ("192.0.2.10/32",)
    }
    assert any("publicIPAddresses/tee-worker-ip" in str(request.url) for request in seen)
    client.destroy(instance)
    assert client.remaining(instance) is False
    assert client.charges(instance) == 2.5


def test_google_confidential_lifecycle() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/zones"):
            return httpx.Response(200, json={"items": [{"name": "zone-a", "status": "UP"}]})
        if path.endswith("/machineTypes/a3-highgpu-1g"):
            return httpx.Response(200, json={"name": "a3-highgpu-1g"})
        if path.endswith("/queries"):
            return httpx.Response(200, json={"rows": [{"f": [{"v": "2.5"}]}]})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.method == "GET" and "global/firewalls" in path:
            return httpx.Response(404, json={})
        if request.method == "GET" and "instances/tee-worker" in path:
            return httpx.Response(404, json={})
        return httpx.Response(200, json={"name": "tee-worker", "status": "RUNNING"})

    client = GoogleConfidentialClient(
        project_id="project",
        billing_export_table="project.billing.export",
        credential=Credential(),
        client=httpx.Client(
            base_url="https://compute.googleapis.com/compute/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    offer = client.offers()[0]
    assert offer["tee_type"] == "tdx"
    instance = client.create(
        offer,
        image="ubuntu-2204-lts",
        disk_gb=64,
        label="tee-worker",
        ssh_public_key="ssh-ed25519 AAA",
        allowed_source_ranges=["192.0.2.10/32"],
    )
    assert instance == "zones/zone-a/instances/tee-worker"
    payload = next(
        json.loads(request.content)
        for request in seen
        if request.method == "POST" and str(request.url).endswith("/instances")
    )
    assert payload["confidentialInstanceConfig"]["confidentialInstanceType"] == "TDX"
    assert payload["scheduling"]["provisioningModel"] == "SPOT"
    assert payload["shieldedInstanceConfig"]["enableSecureBoot"] is True
    assert payload["tags"]["items"] == ["xurcroh-tee"]
    firewall = next(
        json.loads(request.content)
        for request in seen
        if request.method == "POST" and "global/firewalls" in str(request.url)
    )
    assert firewall["sourceRanges"] == ["192.0.2.10/32"]
    assert firewall["targetTags"] == ["xurcroh-tee"]
    client.destroy(instance)
    assert client.remaining(instance) is False
    assert client.charges(instance) == 2.5


def test_google_billing_uses_bound_parameters() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"rows": []})

    client = GoogleConfidentialClient(
        project_id="project",
        billing_export_table="project.billing.export",
        credential=Credential(),
        client=httpx.Client(
            base_url="https://compute.googleapis.com/compute/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    hostile = "zones/zone-a/instances/x' OR 1=1 --"
    assert client.charges(hostile) == 0.0
    payload = json.loads(seen[0].content)
    assert hostile not in payload["query"]
    assert payload["queryParameters"] == [
        {
            "name": "instance_label",
            "parameterType": {"type": "STRING"},
            "parameterValue": {"value": "x' OR 1=1 __"},
        }
    ]
    assert payload["useLegacySql"] is False
