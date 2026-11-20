from __future__ import annotations

import hmac
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cbor2
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from xurcroh.errors import ProviderError
from xurcroh.providers.cloud import instance_label


class AwsNitroClient:
    def __init__(
        self,
        *,
        region_id: str | None = None,
        instance_type: str = "m5.xlarge",
        ec2_client: Any = None,
        kms_client: Any = None,
        cost_client: Any = None,
        nitro_root_certificate: bytes | None = None,
    ) -> None:
        self.region_id = region_id or os.environ.get("AWS_REGION", "us-east-1")
        self.instance_type = instance_type
        if ec2_client is None or kms_client is None or cost_client is None:
            import boto3

            ec2_client = ec2_client or boto3.client("ec2", region_name=self.region_id)
            kms_client = kms_client or boto3.client("kms", region_name=self.region_id)
            cost_client = cost_client or boto3.client("ce", region_name="us-east-1")
        self._ec2 = ec2_client
        self._kms = kms_client
        self._cost = cost_client
        if nitro_root_certificate is None:
            root_path = os.environ.get("AWS_NITRO_ROOT_CERTIFICATE_PATH")
            if root_path:
                nitro_root_certificate = Path(root_path).read_bytes()
        self._nitro_root_certificate = nitro_root_certificate

    @staticmethod
    def _call(operation: str, function: Any, **values: Any) -> Any:
        try:
            return function(**values)
        except Exception as error:
            raise ProviderError(f"AWS {operation} failed") from error

    def offers(self, gpu_name: str | None = None) -> list[dict[str, Any]]:
        if gpu_name:
            return []
        document = self._call(
            "instance-type discovery",
            self._ec2.describe_instance_types,
            InstanceTypes=[self.instance_type],
        )
        offers = []
        for instance in document.get("InstanceTypes", []):
            if not instance.get("EnclaveOptions", {}).get("Enabled"):
                continue
            offers.append(
                {
                    "id": instance["InstanceType"],
                    "provider_id": "aws",
                    "tee_type": "nitro",
                    "gpu_model": None,
                    "gpu_memory_bytes": 0,
                    "cpu_vendor": "Intel",
                    "country": self.region_id,
                    "attested": False,
                    "attestation_required": True,
                    "network": "af-vsock",
                    "managed": True,
                }
            )
        return offers

    def create(self, offer: dict[str, Any], **values: Any) -> str:
        label = instance_label(str(values["label"]))
        instance_type = str(offer["id"])
        if instance_type != self.instance_type:
            raise ValueError("AWS offer does not match the configured Nitro parent type")
        security_groups = values.get("security_group_ids")
        if not isinstance(security_groups, list) or not security_groups:
            raise ValueError("scoped parent security groups are required")
        document = self._call(
            "parent creation",
            self._ec2.run_instances,
            ImageId=str(values["image_id"]),
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=str(values["key_name"]),
            SubnetId=str(values["subnet_id"]),
            SecurityGroupIds=security_groups,
            ClientToken=label,
            EnclaveOptions={"Enabled": True},
            HibernationOptions={"Configured": False},
            MetadataOptions={
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 2,
            },
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": label},
                        {"Key": "xurcroh:managed", "Value": "true"},
                        {"Key": "xurcroh:instance", "Value": label},
                    ],
                }
            ],
        )
        instances = document.get("Instances", [])
        if len(instances) != 1 or not instances[0].get("InstanceId"):
            raise ProviderError("AWS did not return exactly one parent instance")
        return str(instances[0]["InstanceId"])

    def destroy(self, instance_id: str) -> None:
        self._call("parent termination", self._ec2.terminate_instances, InstanceIds=[instance_id])

    def remaining(self, instance_id: str) -> bool:
        try:
            document = self._ec2.describe_instances(InstanceIds=[instance_id])
        except Exception as error:
            response = getattr(error, "response", {})
            if response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                return False
            raise ProviderError("AWS parent lookup failed") from error
        for reservation in document.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                if instance.get("InstanceId") == instance_id:
                    return instance.get("State", {}).get("Name") not in {
                        "shutting-down",
                        "terminated",
                    }
        return False

    def charges(self, instance_id: str) -> float:
        now = datetime.now(UTC)
        total = 0.0
        token = None
        for _ in range(1000):
            query = {
                "TimePeriod": {
                    "Start": now.date().replace(day=1).isoformat(),
                    "End": (now.date() + timedelta(days=1)).isoformat(),
                },
                "Granularity": "DAILY",
                "Metrics": ["UnblendedCost"],
                "Filter": {"Dimensions": {"Key": "RESOURCE_ID", "Values": [instance_id]}},
                "GroupBy": [{"Type": "DIMENSION", "Key": "RESOURCE_ID"}],
            }
            if token is not None:
                query["NextPageToken"] = token
            document = self._call(
                "cost query", self._cost.get_cost_and_usage_with_resources, **query
            )
            for period in document.get("ResultsByTime", []):
                for group in period.get("Groups", []):
                    if instance_id not in group.get("Keys", []):
                        continue
                    cost = group.get("Metrics", {}).get("UnblendedCost", {})
                    if cost.get("Unit") != "USD":
                        raise ProviderError("AWS cost result is not denominated in USD")
                    total += float(cost.get("Amount", 0.0))
            token = document.get("NextPageToken")
            if not token:
                return total
        raise ProviderError("AWS cost pagination did not terminate")

    @staticmethod
    def build_enclave_command(
        *,
        eif_path: str,
        cpu_count: int,
        memory_mib: int,
        enclave_cid: int,
        debug_mode: bool = False,
    ) -> list[str]:
        if debug_mode:
            raise ValueError("debug mode invalidates Nitro attestation measurements")
        if not eif_path.startswith("/") or "\x00" in eif_path:
            raise ValueError("enclave image must be an absolute path")
        if cpu_count < 1 or memory_mib < 128 or enclave_cid < 3:
            raise ValueError("invalid enclave CPU, memory, or CID configuration")
        return [
            "nitro-cli",
            "run-enclave",
            "--eif-path",
            eif_path,
            "--cpu-count",
            str(cpu_count),
            "--memory",
            str(memory_mib),
            "--enclave-cid",
            str(enclave_cid),
        ]

    def _root_certificate(self) -> x509.Certificate:
        if self._nitro_root_certificate is None:
            raise ProviderError("the pinned AWS Nitro root certificate is required")
        try:
            if self._nitro_root_certificate.startswith(b"-----BEGIN"):
                return x509.load_pem_x509_certificate(self._nitro_root_certificate)
            return x509.load_der_x509_certificate(self._nitro_root_certificate)
        except ValueError as error:
            raise ProviderError("AWS Nitro root certificate is invalid") from error

    @staticmethod
    def _verify_certificate_chain(
        leaf: x509.Certificate, cabundle: list[bytes], root: x509.Certificate
    ) -> None:
        if not cabundle:
            raise ProviderError("Nitro attestation CA bundle is empty")
        try:
            bundle = [x509.load_der_x509_certificate(item) for item in cabundle]
        except ValueError as error:
            raise ProviderError("Nitro attestation CA bundle is invalid") from error
        if not hmac.compare_digest(
            bundle[0].public_bytes(serialization.Encoding.DER),
            root.public_bytes(serialization.Encoding.DER),
        ):
            raise ProviderError("Nitro attestation root does not match the pinned root")
        chain = [leaf, *reversed(bundle)]
        now = datetime.now(UTC)
        for index, certificate in enumerate(chain):
            if not certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc:
                raise ProviderError("Nitro attestation certificate is outside its validity period")
            constraints = certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            ).value
            if index == 0 and constraints.ca:
                raise ProviderError("Nitro attestation leaf must not be a CA")
            if index > 0 and not constraints.ca:
                raise ProviderError("Nitro attestation issuer is not a CA")
            if index + 1 >= len(chain):
                continue
            issuer = chain[index + 1]
            if certificate.issuer != issuer.subject:
                raise ProviderError("Nitro attestation certificate chain is not linked")
            algorithm = certificate.signature_hash_algorithm
            parameters = certificate.signature_algorithm_parameters
            if algorithm is None:
                raise ProviderError("Nitro certificate hash algorithm is missing")
            try:
                issuer_key = issuer.public_key()
                if isinstance(issuer_key, ec.EllipticCurvePublicKey):
                    issuer_key.verify(
                        certificate.signature,
                        certificate.tbs_certificate_bytes,
                        ec.ECDSA(algorithm),
                    )
                elif isinstance(issuer_key, rsa.RSAPublicKey) and isinstance(
                    parameters, padding.AsymmetricPadding
                ):
                    issuer_key.verify(
                        certificate.signature,
                        certificate.tbs_certificate_bytes,
                        parameters,
                        algorithm,
                    )
                else:
                    raise ProviderError("Nitro certificate key algorithm is unsupported")
            except (InvalidSignature, TypeError, ValueError) as error:
                raise ProviderError("Nitro attestation certificate signature is invalid") from error

    def verify_attestation_document(
        self,
        attestation_document: bytes,
        *,
        expected_pcrs: dict[int, bytes],
        expected_nonce: bytes,
        expected_public_key: bytes,
    ) -> dict[str, Any]:
        if len(attestation_document) > 65_536:
            raise ProviderError("Nitro attestation document is too large")
        if not {0, 1, 2}.issubset(expected_pcrs):
            raise ProviderError("expected Nitro PCRs must include image registers 0, 1, and 2")
        try:
            cose = cbor2.loads(attestation_document)
        except cbor2.CBORError as error:
            raise ProviderError("Nitro attestation CBOR is invalid") from error
        if isinstance(cose, cbor2.CBORTag):
            if cose.tag != 18:
                raise ProviderError("Nitro attestation is not COSE_Sign1")
            cose = cose.value
        if not isinstance(cose, list) or len(cose) != 4:
            raise ProviderError("Nitro COSE_Sign1 structure is invalid")
        protected, unprotected, payload, signature = cose
        if not isinstance(protected, bytes) or not isinstance(payload, bytes):
            raise ProviderError("Nitro COSE protected fields are invalid")
        try:
            headers = cbor2.loads(protected)
            document = cbor2.loads(payload)
        except cbor2.CBORError as error:
            raise ProviderError("Nitro COSE content is invalid") from error
        if (
            headers != {1: -35}
            or unprotected != {}
            or not isinstance(signature, bytes)
            or len(signature) != 96
        ):
            raise ProviderError("Nitro attestation must use ES384")
        if not isinstance(document, dict):
            raise ProviderError("Nitro attestation payload is invalid")
        required_bytes = ("certificate", "cabundle", "pcrs", "nonce", "public_key")
        if any(not isinstance(document.get(name), (bytes, dict, list)) for name in required_bytes):
            raise ProviderError("Nitro attestation required fields are missing")
        if document.get("digest") != "SHA384" or not document.get("module_id"):
            raise ProviderError("Nitro attestation digest or module is invalid")
        timestamp = document.get("timestamp")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool):
            raise ProviderError("Nitro attestation timestamp is invalid")
        now_ms = int(time.time() * 1000)
        if timestamp > now_ms + 60_000 or timestamp < now_ms - 300_000:
            raise ProviderError("Nitro attestation is stale or not yet valid")
        if not isinstance(document["nonce"], bytes) or not hmac.compare_digest(
            document["nonce"], expected_nonce
        ):
            raise ProviderError("Nitro attestation nonce does not match")
        if not isinstance(document["public_key"], bytes) or not hmac.compare_digest(
            document["public_key"], expected_public_key
        ):
            raise ProviderError("Nitro attestation session key does not match")
        pcrs = document["pcrs"]
        if not isinstance(pcrs, dict):
            raise ProviderError("Nitro attestation PCR map is invalid")
        for index, expected in expected_pcrs.items():
            actual = pcrs.get(index)
            if not isinstance(actual, bytes) or not hmac.compare_digest(actual, expected):
                raise ProviderError(f"Nitro attestation PCR {index} does not match")
        try:
            leaf = x509.load_der_x509_certificate(document["certificate"])
        except (TypeError, ValueError) as error:
            raise ProviderError("Nitro attestation leaf certificate is invalid") from error
        cabundle = document["cabundle"]
        if not isinstance(cabundle, list) or not all(isinstance(item, bytes) for item in cabundle):
            raise ProviderError("Nitro attestation CA bundle is invalid")
        self._verify_certificate_chain(leaf, cabundle, self._root_certificate())
        public_key = leaf.public_key()
        if not isinstance(public_key, ec.EllipticCurvePublicKey) or not isinstance(
            public_key.curve, ec.SECP384R1
        ):
            raise ProviderError("Nitro attestation leaf key is not P-384")
        try:
            r = int.from_bytes(signature[:48], "big")
            s = int.from_bytes(signature[48:], "big")
            sig_structure = cbor2.dumps(["Signature1", protected, b"", payload])
            public_key.verify(encode_dss_signature(r, s), sig_structure, ec.ECDSA(hashes.SHA384()))
        except (InvalidSignature, TypeError) as error:
            raise ProviderError("Nitro attestation COSE signature is invalid") from error
        return document

    def decrypt_with_attestation(
        self,
        *,
        key_id: str,
        ciphertext: bytes,
        attestation_document: bytes,
        expected_pcrs: dict[int, bytes],
        expected_nonce: bytes,
        expected_public_key: bytes,
    ) -> bytes:
        self.verify_attestation_document(
            attestation_document,
            expected_pcrs=expected_pcrs,
            expected_nonce=expected_nonce,
            expected_public_key=expected_public_key,
        )
        response = self._call(
            "attested KMS decryption",
            self._kms.decrypt,
            KeyId=key_id,
            CiphertextBlob=ciphertext,
            Recipient={
                "KeyEncryptionAlgorithm": "RSAES_OAEP_SHA_256",
                "AttestationDocument": attestation_document,
            },
        )
        plaintext = response.get("Plaintext")
        if not isinstance(plaintext, bytes):
            raise ProviderError("AWS KMS did not return attested plaintext")
        return plaintext
