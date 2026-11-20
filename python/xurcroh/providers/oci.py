from __future__ import annotations

import os
import re
import secrets
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt

from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import instance_label

SUPPORTED_SHAPES = {"VM.Standard.E5.Flex", "VM.Standard.E6.Flex"}


def _https_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("attestation endpoints must be credential-free HTTPS URLs")
    return value.rstrip("/")


def _reference_measurements(document: Any) -> Iterator[str]:
    if isinstance(document, dict):
        for key, value in document.items():
            if key == "value" and isinstance(value, str):
                yield value
            else:
                yield from _reference_measurements(value)
    elif isinstance(document, list):
        for value in document:
            yield from _reference_measurements(value)


class OciConfidentialClient:
    def __init__(
        self,
        *,
        tenancy_id: str | None = None,
        compartment_id: str | None = None,
        config: dict[str, Any] | None = None,
        compute_client: Any = None,
        usage_client: Any = None,
        attestation_client: httpx.Client | None = None,
        verifier_public_key_pem: bytes | None = None,
        attestation_policy_id: str | None = None,
    ) -> None:
        if config is None and (compute_client is None or usage_client is None):
            import oci

            config = oci.config.from_file()
        config = config or {}
        self.tenancy_id = tenancy_id or os.environ.get("OCI_TENANCY_OCID") or config.get("tenancy")
        self.compartment_id = (
            compartment_id
            or os.environ.get("OCI_COMPARTMENT_OCID")
            or config.get("compartment_id")
            or self.tenancy_id
        )
        if not self.tenancy_id or not self.compartment_id:
            raise ProviderError("OCI tenancy and compartment OCIDs are required")
        if compute_client is None or usage_client is None:
            import oci

            compute_client = compute_client or oci.core.ComputeClient(config)
            usage_client = usage_client or oci.usage_api.UsageapiClient(config)
        self._compute = compute_client
        self._usage = usage_client
        self._attestation_http = attestation_client or httpx.Client(timeout=60)
        if verifier_public_key_pem is None:
            key_path = os.environ.get("XURCROH_VERAISON_PUBLIC_KEY_PATH")
            if key_path:
                verifier_public_key_pem = Path(key_path).read_bytes()
        self._verifier_public_key = verifier_public_key_pem
        self._attestation_policy_id = attestation_policy_id or os.environ.get(
            "XURCROH_VERAISON_POLICY_ID", "policy:SEVSNP"
        )

    @staticmethod
    def _call(operation: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            raise ProviderError(f"OCI {operation} failed") from error

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name:
            return []
        result = self._call(
            "shape discovery",
            self._compute.list_shapes,
            compartment_id=self.compartment_id,
        )
        return [
            {
                "id": shape.shape,
                "provider_id": "oci",
                "tee_type": "sev-snp",
                "gpu_model": None,
                "gpu_memory_bytes": 0,
                "cpu_vendor": "AMD",
                "country": None,
                "attested": False,
                "attestation_required": True,
                "managed": True,
            }
            for shape in result.data
            if shape.shape in SUPPORTED_SHAPES and not shape.gpus
        ]

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        import oci

        label = instance_label(str(values["label"]))
        shape = str(offer["id"])
        if shape not in SUPPORTED_SHAPES:
            raise ValueError("OCI confidential instances require E5 or E6 Flex shapes")
        public_key = values.get("ssh_public_key")
        if not isinstance(public_key, str) or not public_key.startswith(
            ("ssh-ed25519 ", "ssh-rsa ")
        ):
            raise ValueError("an SSH public key is required")
        nsg_ids = values.get("network_security_group_ids")
        if (
            not isinstance(nsg_ids, list)
            or not nsg_ids
            or not all(isinstance(item, str) and item for item in nsg_ids)
        ):
            raise ValueError("at least one scoped network security group is required")
        details = oci.core.models.LaunchInstanceDetails(
            compartment_id=self.compartment_id,
            availability_domain=str(values["availability_domain"]),
            display_name=label,
            shape=shape,
            shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                ocpus=float(values.get("ocpus", 4)),
                memory_in_gbs=float(values.get("memory_gb", 64)),
            ),
            platform_config=oci.core.models.LaunchInstancePlatformConfig(
                type="AMD_VM",
                is_memory_encryption_enabled=True,
            ),
            source_details=oci.core.models.InstanceSourceViaImageDetails(
                source_type="image",
                image_id=str(values["image_id"]),
                boot_volume_size_in_gbs=int(values.get("disk_gb", 64)),
            ),
            create_vnic_details=oci.core.models.CreateVnicDetails(
                subnet_id=str(values["subnet_id"]),
                assign_public_ip=bool(values.get("assign_public_ip", True)),
                nsg_ids=nsg_ids,
            ),
            metadata={"ssh_authorized_keys": public_key},
            freeform_tags={"xurcroh-managed": "true", "xurcroh-instance": label},
        )
        result = self._call("instance creation", self._compute.launch_instance, details)
        instance_id = result.data.id
        if not isinstance(instance_id, str) or not instance_id:
            raise ProviderError("OCI did not return an instance OCID")
        return instance_id

    def destroy(self, instance_id: str) -> None:
        self._call("instance termination", self._compute.terminate_instance, instance_id)

    def remaining(self, instance_id: str) -> bool:
        try:
            result = self._compute.get_instance(instance_id)
        except Exception as error:
            status = getattr(error, "status", None)
            if status == 404:
                return False
            raise ProviderError("OCI instance lookup failed") from error
        return result.data.lifecycle_state not in {"TERMINATED", "TERMINATING"}

    def charges(self, instance_id: str) -> float:
        import oci

        instance = self._call("instance lookup", self._compute.get_instance, instance_id).data
        label = instance.freeform_tags.get("xurcroh-instance")
        if not label:
            raise ProviderError("OCI instance lacks its Xurcroh billing tag")
        now = datetime.now(UTC)
        details = oci.usage_api.models.RequestSummarizedUsagesDetails(
            tenant_id=self.tenancy_id,
            time_usage_started=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            time_usage_ended=now,
            granularity="DAILY",
            query_type="COST",
            group_by=["resourceId"],
            filter=oci.usage_api.models.Filter(
                tags=[oci.usage_api.models.Tag(key="xurcroh-instance", value=label)]
            ),
        )
        result = self._call("usage query", self._usage.request_summarized_usages, details)
        total = 0.0
        for item in result.data.items:
            if item.resource_id != instance_id:
                continue
            if item.currency not in {None, "USD"}:
                raise ProviderError("non-USD OCI bills require explicit FX conversion")
            total += float(item.computed_amount or 0.0)
        return total

    def _json_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError("Veraison challenge-response request failed") from error
        if not isinstance(document, dict):
            raise ProviderError("Veraison response must be a JSON object")
        return document

    def attest(
        self,
        *,
        attester_url: str,
        verifier_url: str,
        expected_measurements: list[str],
    ) -> dict[str, Any]:
        if self._verifier_public_key is None:
            raise ProviderError("a pinned Veraison verifier public key is required")
        if not expected_measurements or not all(expected_measurements):
            raise ProviderError("at least one expected SEV-SNP measurement is required")
        attester_url = _https_url(attester_url)
        verifier_url = _https_url(verifier_url)
        nonce = secrets.token_urlsafe(64)
        session_response = self._attestation_http.post(
            f"{verifier_url}/challenge-response/v1/newSession",
            params={"nonce": nonce},
        )
        session = self._json_response(session_response)
        location = session_response.headers.get("location") or session.get("location")
        if session.get("nonce") != nonce:
            raise ProviderError("Veraison session nonce mismatch")
        if (
            not isinstance(location, str)
            or re.fullmatch(r"session/[A-Za-z0-9-]+", location) is None
        ):
            location = session.get("session") or session.get("id")
        if not isinstance(location, str):
            raise ProviderError("Veraison session location is missing")
        if re.fullmatch(r"(?:session/)?[A-Za-z0-9-]+", location) is None:
            raise ProviderError("Veraison session location is invalid")
        evidence = self._json_response(
            self._attestation_http.post(
                f"{attester_url}/ratsd/chares",
                json={"nonce": nonce},
            )
        )
        if evidence.get("eat_nonce") != nonce or not evidence.get("cmw"):
            raise ProviderError("SEV-SNP evidence is not bound to the session nonce")
        result = self._json_response(
            self._attestation_http.post(
                f"{verifier_url}/challenge-response/v1/{location}",
                json=evidence,
                headers={
                    "Content-Type": (
                        'application/eat+cwt; profile="tag:github.com,2025:veraison/ratsd/cmw"'
                    ),
                    "Accept": "application/vnd.veraison.challenge-response-session+json",
                },
            )
        )
        if result.get("status") != "complete" or result.get("nonce") != nonce:
            raise ProviderError("Veraison appraisal is incomplete or replayed")
        token = result.get("result")
        if not isinstance(token, str):
            raise ProviderError("Veraison appraisal token is missing")
        try:
            claims = jwt.decode(
                token,
                self._verifier_public_key,
                algorithms=["ES256"],
                options={"verify_aud": False, "require": ["iat"]},
            )
        except (jwt.InvalidTokenError, ValueError) as error:
            raise ProviderError("Veraison appraisal signature is invalid") from error
        issued_at = claims.get("iat")
        if not isinstance(issued_at, (int, float)) or isinstance(issued_at, bool):
            raise ProviderError("Veraison appraisal timestamp is invalid")
        if issued_at > time.time() + 60 or issued_at < time.time() - 300:
            raise ProviderError("Veraison appraisal is stale or not yet valid")
        if claims.get("eat_nonce") != nonce:
            raise ProviderError("Veraison appraisal nonce mismatch")
        submods = claims.get("submods")
        snp = submods.get("SEVSNP") if isinstance(submods, dict) else None
        if not isinstance(snp, dict):
            raise ProviderError("Veraison result has no SEV-SNP appraisal")
        if snp.get("ear.status") != "affirming":
            raise ProviderError("SEV-SNP appraisal is not affirming")
        if snp.get("ear.appraisal-policy-id") != self._attestation_policy_id:
            raise ProviderError("SEV-SNP appraisal used an unexpected policy")
        vector = snp.get("ear.trustworthiness-vector")
        if not isinstance(vector, dict) or any(
            vector.get(name) != 2 for name in ("hardware", "runtime-opaque")
        ):
            raise ProviderError("SEV-SNP hardware or runtime opacity was not affirmed")
        reference_values = snp.get("ear.veraison.reference-values")
        if not isinstance(reference_values, (dict, list)) or not reference_values:
            raise ProviderError("SEV-SNP appraisal has no reference values")
        available = set(_reference_measurements(reference_values))
        if not set(expected_measurements).issubset(available):
            raise ProviderError("SEV-SNP workload measurements do not match")
        return claims
