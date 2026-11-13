from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class LadderEvidenceError(ValueError):
    pass


def _string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise LadderEvidenceError(f"{key} must be a non-empty string")
    return value


def _integer(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LadderEvidenceError(f"{key} must be an integer")
    return value


def _hex(value: str, length: int, name: str) -> None:
    if len(value) != length or re.fullmatch(r"[0-9a-f]+", value) is None:
        raise LadderEvidenceError(f"{name} must be {length} lowercase hexadecimal characters")


@dataclass(frozen=True)
class WorkerPlacement:
    name: str
    endpoint: str
    machine_id: int
    provider_id: str
    country: str
    cpu_vendor: str
    activation_index: int
    weight_index: int
    recipient_public: str

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> WorkerPlacement:
        worker = cls(
            name=_string(document, "name"),
            endpoint=_string(document, "endpoint"),
            machine_id=_integer(document, "machine_id"),
            provider_id=_string(document, "provider_id"),
            country=_string(document, "country"),
            cpu_vendor=_string(document, "cpu_vendor"),
            activation_index=_integer(document, "activation_index"),
            weight_index=_integer(document, "weight_index"),
            recipient_public=_string(document, "recipient_public"),
        )
        if worker.machine_id < 1 or worker.activation_index < 0 or worker.weight_index < 0:
            raise LadderEvidenceError("worker machine and role indexes are invalid")
        _hex(worker.recipient_public, 64, "recipient_public")
        return worker


@dataclass(frozen=True)
class LadderRung:
    rung: str
    model: str
    modality: str
    repository: str
    revision: str
    license: str
    checkpoint_fingerprint: str
    share_count: int
    workers: tuple[WorkerPlacement, ...]

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> LadderRung:
        raw_workers = document.get("workers")
        if not isinstance(raw_workers, list):
            raise LadderEvidenceError("workers must be a list")
        rung = cls(
            rung=_string(document, "rung"),
            model=_string(document, "model"),
            modality=_string(document, "modality"),
            repository=_string(document, "repository"),
            revision=_string(document, "revision"),
            license=_string(document, "license"),
            checkpoint_fingerprint=_string(document, "checkpoint_fingerprint"),
            share_count=_integer(document, "share_count"),
            workers=tuple(WorkerPlacement.from_mapping(worker) for worker in raw_workers),
        )
        rung._validate()
        return rung

    def _validate(self) -> None:
        if re.fullmatch(r"(F0|M0|L[0-6]|M[1-4])", self.rung) is None:
            raise LadderEvidenceError("unknown ladder rung")
        if self.modality not in {"text", "vlm", "image", "video"}:
            raise LadderEvidenceError("unsupported ladder modality")
        _hex(self.revision, 40, "checkpoint revision")
        _hex(self.checkpoint_fingerprint, 64, "checkpoint_fingerprint")
        if self.share_count < 2:
            raise LadderEvidenceError("at least two activation and weight shares are required")
        expected = {
            (activation, weight)
            for activation in range(self.share_count)
            for weight in range(self.share_count)
        }
        actual = {(worker.activation_index, worker.weight_index) for worker in self.workers}
        if actual != expected or len(self.workers) != len(expected):
            raise LadderEvidenceError("worker roles must form one complete share grid")
        for field in ("name", "machine_id"):
            values = [getattr(worker, field) for worker in self.workers]
            if len(set(values)) != len(values):
                raise LadderEvidenceError(f"worker {field} values must be unique")
        for field in ("provider_id", "country", "cpu_vendor"):
            if len({getattr(worker, field) for worker in self.workers}) < 2:
                raise LadderEvidenceError(f"worker {field} diversity is required")


def _metric(report: Mapping[str, Any], name: str) -> float:
    value = report.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LadderEvidenceError(f"{name} must be numeric")
    metric = float(value)
    if not math.isfinite(metric):
        raise LadderEvidenceError(f"{name} must be finite")
    return metric


def validate_rung_evidence(rung: LadderRung, report: Mapping[str, Any]) -> None:
    hardware = rung.rung not in {"F0", "M0"}
    if report.get("hardware_validated") is not hardware:
        raise LadderEvidenceError("hardware validation status is inconsistent with the rung")
    if report.get("production_accepted") is not False:
        raise LadderEvidenceError("a rung cannot self-declare production acceptance")
    if _metric(report, "quantized_reference_max_abs_error") != 0:
        raise LadderEvidenceError("quantized reference output must match exactly")
    if _metric(report, "original_model_max_abs_error") < 0:
        raise LadderEvidenceError("original-model quality error cannot be negative")
    if not isinstance(report.get("attack_results"), dict):
        raise LadderEvidenceError("attack results are required")
    if _metric(report, "freivalds_failures") != 0:
        raise LadderEvidenceError("Freivalds failures are not allowed")
    if report.get("canary_tripped") is not True:
        raise LadderEvidenceError("canary attribution was not exercised")
    exit_codes = report.get("worker_exit_codes")
    if (
        not isinstance(exit_codes, list)
        or len(exit_codes) != len(rung.workers)
        or any(isinstance(code, bool) or code != 0 for code in exit_codes)
    ):
        raise LadderEvidenceError("every worker must exit successfully")
    if _metric(report, "bytes_sent") <= 0 or _metric(report, "bytes_received") <= 0:
        raise LadderEvidenceError("measured network traffic is required")
    if _metric(report, "inference_seconds") <= 0:
        raise LadderEvidenceError("measured inference time is required")
    if _metric(report, "cost_usd") < 0:
        raise LadderEvidenceError("cost cannot be negative")
    destruction = report.get("destruction")
    if (
        not isinstance(destruction, dict)
        or destruction.get("all_destroyed") is not True
        or destruction.get("remaining_instance_ids") != []
    ):
        raise LadderEvidenceError("verified teardown is incomplete")


def run_ladder_rung(
    rung: LadderRung,
    *,
    execute: Callable[[LadderRung], Mapping[str, Any]],
    teardown: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        execution = execute(rung)
    except BaseException:
        teardown()
        raise
    report = {**execution, "destruction": dict(teardown())}
    validate_rung_evidence(rung, report)
    return report


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def write_rung_evidence(
    path: str | Path,
    rung: LadderRung,
    report: Mapping[str, Any],
    *,
    signing_key: Ed25519PrivateKey,
) -> None:
    validate_rung_evidence(rung, report)
    document = {
        "format": "xurcroh-rung-evidence-v1",
        "rung": asdict(rung),
        "report": dict(report),
    }
    document["signature"] = signing_key.sign(_canonical(document)).hex()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def verify_rung_evidence(
    path: str | Path, *, public_key: Ed25519PublicKey
) -> tuple[LadderRung, dict[str, Any]]:
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or document.get("format") != "xurcroh-rung-evidence-v1":
        raise LadderEvidenceError("unsupported rung evidence format")
    signature = document.pop("signature", None)
    if not isinstance(signature, str):
        raise LadderEvidenceError("rung evidence signature is missing")
    try:
        public_key.verify(bytes.fromhex(signature), _canonical(document))
    except (InvalidSignature, ValueError) as error:
        raise LadderEvidenceError("rung evidence signature is invalid") from error
    rung = LadderRung.from_mapping(document["rung"])
    report = document["report"]
    if not isinstance(report, dict):
        raise LadderEvidenceError("rung report must be an object")
    validate_rung_evidence(rung, report)
    return rung, report
