from __future__ import annotations

from types import SimpleNamespace

import pytest

from xurcroh.attestation import AttestationEvidence
from xurcroh.errors import AttestationError
from xurcroh.providers.ibm import IbmConfidentialClient


class DetailedResponse:
    def __init__(self, result) -> None:
        self._result = result

    def get_result(self):
        return self._result


class FakeVpc:
    def __init__(self) -> None:
        self.calls = []

    def list_instance_profiles(self, **values):
        self.calls.append(values)
        return DetailedResponse(
            {
                "profiles": [
                    {
                        "name": "bx3d-2x10",
                        "status": "current",
                        "confidential_compute_modes": {
                            "default": "tdx",
                            "values": ["disabled", "tdx"],
                        },
                        "gpu_count": {"value": 0},
                        "zones": [{"name": "us-south-1"}],
                    },
                    {
                        "name": "gx3-16x80x1",
                        "status": "current",
                        "confidential_compute_modes": {
                            "default": "tdx",
                            "values": ["disabled", "tdx"],
                        },
                        "gpu_count": {"value": 1},
                        "zones": [{"name": "us-south-1"}],
                    },
                ]
            }
        )

    def create_instance(self, prototype):
        self.calls.append(prototype)
        return DetailedResponse({"id": "instance-id", "status": "running"})

    def get_instance(self, id):
        self.calls.append(id)
        return DetailedResponse(
            {
                "id": id,
                "status": "running",
                "resource_instance_id": "resource-instance-guid",
            }
        )

    def delete_instance(self, id):
        self.calls.append(id)
        return DetailedResponse({})


class FakeUsage:
    def __init__(self) -> None:
        self.calls = []

    def get_resource_usage_account(self, **values):
        self.calls.append(values)
        return DetailedResponse(
            {
                "resources": [
                    {
                        "resource_instance_id": "resource-instance-guid",
                        "currency_code": "USD",
                        "usage": [
                            {"metric": "CPU", "cost": 1.25},
                            {"metric": "INFORMATION", "cost": 99, "non_chargeable": True},
                        ],
                    }
                ]
            }
        )


def client(vpc=None, usage=None):
    return IbmConfidentialClient(
        account_id="account-test",
        region_id="us-south",
        vpc_client=vpc or FakeVpc(),
        usage_client=usage or FakeUsage(),
    )


def test_ibm_tdx_lifecycle_and_usage() -> None:
    vpc = FakeVpc()
    usage = FakeUsage()
    ibm = client(vpc, usage)
    offer = ibm.offers()[0]
    assert offer["id"] == "bx3d-2x10"
    assert offer["tee_type"] == "tdx"
    assert offer["gpu_memory_bytes"] == 0
    assert len(ibm.offers()) == 1
    instance = ibm.create(
        offer,
        label="tee-worker",
        image_id="image-id",
        vpc_id="vpc-id",
        subnet_id="subnet-id",
        security_group_ids=["sg-id"],
        ssh_key_ids=["key-id"],
    )
    assert instance == "instance-id"
    prototype = vpc.calls[-1]
    assert prototype.confidential_compute_mode == "tdx"
    assert prototype.enable_secure_boot is True
    assert prototype.metadata_service.protocol == "https"
    assert prototype.primary_network_interface.security_groups[0].id == "sg-id"
    assert ibm.remaining(instance) is True
    assert ibm.charges(instance) == 1.25
    assert usage.calls[0]["resource_instance_id"] == "resource-instance-guid"
    ibm.destroy(instance)
    assert vpc.calls[-1] == instance


def test_ibm_uses_intel_dcap_verifier(monkeypatch) -> None:
    ibm = client()
    evidence = AttestationEvidence(
        vendor="intel",
        nonce=b"attester-chosen",
        report_data="",
        workload_hash=b"workload",
        artifact_hash=b"artifact",
        public_key_hash=b"key",
    )
    collateral = object()
    called = {}

    def verify(verifier, received_evidence, **values):
        assert received_evidence == evidence
        called["expected_nonce"] = verifier.expected_nonce
        called.update(values)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr("xurcroh.attestation.IntelDCAPVerifier.verify", verify)
    result = ibm.verify_attestation(
        evidence=evidence,
        quote=b"quote",
        collateral=collateral,
        expected_nonce=b"challenger-nonce",
    )
    assert result.ok is True
    assert called == {
        "expected_nonce": b"challenger-nonce",
        "quote": b"quote",
        "collateral": collateral,
    }


def test_ibm_verify_attestation_rejects_nonce_not_matching_challenge() -> None:
    ibm = client()
    evidence = AttestationEvidence(
        vendor="intel",
        nonce=b"attester-chosen",
        report_data="",
        workload_hash=b"workload",
        artifact_hash=b"artifact",
        public_key_hash=b"key",
    )
    with pytest.raises(AttestationError, match="nonce"):
        ibm.verify_attestation(
            evidence=evidence,
            quote=bytes(1024),
            collateral=object(),
            expected_nonce=b"challenger-nonce",
        )
