from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from xurcroh.errors import BudgetExceededError, ProviderError, TeardownError

API_ROOT = "https://console.vast.ai"
ACTIVE_STATES = {None, "", "loading", "running", "stopped", "unknown", "offline"}
TERMINAL_FAILURE_STATES = {"exited", "unknown", "offline"}


@dataclass(frozen=True)
class BudgetPolicy:
    target_usd: float | None = 25.0
    hard_limit_usd: float | None = 30.0
    bandwidth_reserve_gb: float = 2.0
    bandwidth_usd_per_gb: float = 0.05

    @classmethod
    def uncapped(cls) -> BudgetPolicy:
        return cls(target_usd=None, hard_limit_usd=None)

    def __post_init__(self) -> None:
        if self.target_usd is None or self.hard_limit_usd is None:
            if self.target_usd is not None or self.hard_limit_usd is not None:
                raise ValueError("uncapped budgets must omit both target and hard limit")
        elif not 0 < self.target_usd <= self.hard_limit_usd <= 30:
            raise ValueError("budget must satisfy 0 < target <= hard limit <= $30")
        if self.bandwidth_reserve_gb < 0 or self.bandwidth_usd_per_gb < 0:
            raise ValueError("bandwidth allowance cannot be negative")


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "instances": [], "events": [], "watchdog_pid": None}
        document: dict[str, Any] = json.loads(self.path.read_text())
        if document.get("schema_version") != 1 or not isinstance(document.get("instances"), list):
            raise ProviderError("invalid Vast state file")
        return document

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock:
            return self._read()

    def update(self, event: str, mutate: Any = None, **values: Any) -> dict[str, Any]:
        with self._thread_lock:
            descriptor = os.open(self.path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "r+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                document = self._read()
                if mutate is not None:
                    mutate(document)
                if values:
                    document.update(values)
                document.setdefault("events", []).append(
                    {"event": event, "timestamp": time.time(), **values}
                )
                temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
                return document

    def add_instance(self, record: dict[str, Any]) -> None:
        def add(document: dict[str, Any]) -> None:
            if any(item["id"] == record["id"] for item in document["instances"]):
                raise ProviderError("instance is already tracked")
            document["instances"].append(record)

        self.update("instance_created", add, instance_id=record["id"])

    def mark_destroyed(self, instance_id: int, estimated_final_usd: float) -> None:
        def mark(document: dict[str, Any]) -> None:
            for item in document["instances"]:
                if item["id"] == instance_id:
                    item["destroyed_at"] = time.time()
                    item["estimated_final_usd"] = estimated_final_usd
                    return
            raise ProviderError("cannot mark an untracked instance destroyed")

        self.update("instance_destroyed", mark, instance_id=instance_id)

    def record_charges(self, charges: dict[int, float]) -> None:
        def record(document: dict[str, Any]) -> None:
            for item in document["instances"]:
                item["reported_charges_usd"] = charges.get(item["id"], 0.0)
            document["charges_refreshed_at"] = time.time()

        self.update("charges_refreshed", record)


class BudgetGuard:
    def __init__(self, store: StateStore, policy: BudgetPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or BudgetPolicy()

    def instance_estimate(self, record: dict[str, Any], now: float) -> float:
        reported = float(record.get("reported_charges_usd", 0.0))
        end = float(record.get("destroyed_at", now))
        elapsed_hours = max(0.0, end - float(record["created_at"])) / 3600
        hourly = float(record["compute_hourly_usd"]) + float(record["storage_hourly_usd"])
        estimated = hourly * elapsed_hours + float(record.get("bandwidth_usd", 0.0))
        if "estimated_final_usd" in record:
            estimated = float(record["estimated_final_usd"])
        return max(reported, estimated)

    def total(self, snapshot: dict[str, Any] | None = None, now: float | None = None) -> float:
        document = snapshot or self.store.snapshot()
        current = time.time() if now is None else now
        return sum(self.instance_estimate(record, current) for record in document["instances"])

    def forecast(self, offer: dict[str, Any], planned_hours: float, disk_gb: float) -> float:
        if planned_hours <= 0 or disk_gb <= 0:
            raise ValueError("planned hours and disk size must be positive")
        storage_hourly = float(offer.get("storage_total_cost", 0.0))
        if storage_hourly == 0 and offer.get("storage_cost"):
            storage_hourly = float(offer["storage_cost"]) * disk_gb / 720
        hourly = float(offer["dph_total"]) + storage_hourly
        return hourly * planned_hours + (
            self.policy.bandwidth_reserve_gb * self.policy.bandwidth_usd_per_gb
        )

    def require_capacity(self, additional_usd: float) -> None:
        if additional_usd < 0 or not math.isfinite(additional_usd):
            raise ValueError("forecast must be a finite non-negative cost")
        if self.policy.target_usd is None:
            return
        projected = self.total() + additional_usd
        if projected > self.policy.target_usd:
            raise BudgetExceededError(
                f"forecast ${projected:.2f} exceeds the ${self.policy.target_usd:.2f} target"
            )

    def require_below_limit(self) -> None:
        if self.policy.target_usd is None:
            return
        current = self.total()
        if current >= self.policy.target_usd:
            raise BudgetExceededError(
                f"estimated or reported spend ${current:.2f} reached the teardown threshold"
            )


def sanitize_instance(document: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "actual_status",
        "cur_state",
        "status",
        "status_msg",
        "image",
        "label",
        "machine_id",
        "ssh_host",
        "ssh_port",
        "ports",
        "public_ipaddr",
        "gpu_name",
        "gpu_totalram",
        "num_gpus",
        "dph_total",
        "storage_total_cost",
        "disk_space",
        "start_date",
        "geolocation",
    )
    return {field: document.get(field) for field in fields if field in document}


def sanitize_offer(document: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "machine_id",
        "gpu_name",
        "gpu_ram",
        "num_gpus",
        "dph_total",
        "storage_total_cost",
        "storage_cost",
        "disk_space",
        "reliability",
        "geolocation",
        "verification",
        "cpu_vendor",
        "cpu_name",
        "cpu_ram",
        "host_id",
        "provider_id",
    )
    result = {field: document.get(field) for field in fields if field in document}
    cpu_name = str(result.get("cpu_name") or "").lower()
    if not result.get("cpu_vendor"):
        result["cpu_vendor"] = (
            "AMD" if "amd" in cpu_name else "Intel" if "intel" in cpu_name else "unknown"
        )
    host_id = result.get("host_id")
    if not result.get("provider_id") and host_id is not None:
        result["provider_id"] = f"vast-host-{int(host_id)}"
    return result


def select_diverse_offers(
    offers: list[dict[str, Any]],
    *,
    count: int,
    guard: BudgetGuard,
    planned_hours: float,
    disk_gb: float,
    max_country_fraction: float = 0.5,
    max_cpu_vendor_fraction: float = 0.5,
    max_provider_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], float]:
    if (
        count < 1
        or not 0 < max_country_fraction <= 1
        or not 0 < max_cpu_vendor_fraction <= 1
        or not 0 < max_provider_fraction <= 1
    ):
        raise ValueError("invalid host count or diversity fraction")
    country_limit = max(1, int(count * max_country_fraction))
    vendor_limit = max(1, int(count * max_cpu_vendor_fraction))
    provider_limit = max(1, int(count * max_provider_fraction))
    selected: list[dict[str, Any]] = []
    machines: set[int] = set()
    countries: dict[str, int] = {}
    vendors: dict[str, int] = {}
    providers: dict[str, int] = {}
    forecast = 0.0
    for _ in range(count):
        candidates = []
        for offer in offers:
            machine = int(offer["machine_id"])
            country = str(offer.get("geolocation") or "unknown")
            vendor = str(offer.get("cpu_vendor") or "unknown")
            provider = str(offer.get("provider_id") or "vast")
            if (
                machine in machines
                or countries.get(country, 0) >= country_limit
                or vendors.get(vendor, 0) >= vendor_limit
                or providers.get(provider, 0) >= provider_limit
            ):
                continue
            cost = guard.forecast(offer, planned_hours, disk_gb)
            candidates.append(
                (
                    -int(countries.get(country, 0) == 0),
                    -int(vendors.get(vendor, 0) == 0),
                    -int(providers.get(provider, 0) == 0),
                    float(offer["dph_total"]),
                    int(offer["id"]),
                    offer,
                    cost,
                )
            )
        if not candidates:
            raise ProviderError(
                "not enough distinct eligible GPU hosts within the diversity policy"
            )
        _, _, _, _, _, chosen, cost = min(candidates, key=lambda item: item[:5])
        selected.append(chosen)
        machines.add(int(chosen["machine_id"]))
        country = str(chosen.get("geolocation") or "unknown")
        vendor = str(chosen.get("cpu_vendor") or "unknown")
        countries[country] = countries.get(country, 0) + 1
        vendors[vendor] = vendors.get(vendor, 0) + 1
        provider = str(chosen.get("provider_id") or "vast")
        providers[provider] = providers.get(provider, 0) + 1
        forecast += cost
    if count > 1 and (len(countries) < 2 or len(vendors) < 2 or len(providers) < 2):
        raise ProviderError("selected GPU hosts do not satisfy provider diversity")
    guard.require_capacity(forecast)
    return selected, forecast


class VastClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
        base_url: str = API_ROOT,
    ) -> None:
        credential = api_key or os.environ.get("VAST_API_KEY")
        if not credential:
            raise ProviderError("VAST_API_KEY must be supplied through the environment")
        if not base_url.startswith("https://"):
            raise ValueError("Vast API requires HTTPS")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {credential}"},
            timeout=30,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VastClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            for attempt in range(4):
                response = self._client.request(method, path, **kwargs)
                if response.status_code != 429 or attempt == 3:
                    break
                try:
                    delay = min(60.0, float(response.headers.get("retry-after", 5 * 2**attempt)))
                except ValueError:
                    delay = 5 * 2**attempt
                time.sleep(delay)
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPStatusError as error:
            raise ProviderError(f"Vast API returned HTTP {error.response.status_code}") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(
                f"Vast API {method} {path} failed: {type(error).__name__}"
            ) from error
        if not isinstance(document, dict) or document.get("success") is False:
            raise ProviderError(f"Vast API rejected {method} {path}")
        return document

    def offers(self, gpu_name: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "reliability": {"gte": 0.99},
            "num_gpus": {"eq": 1},
            "direct_port_count": {"gte": 1},
            "type": "ondemand",
            "order": [["dph_total", "asc"]],
            "limit": limit,
        }
        if gpu_name is not None:
            query["gpu_name"] = {"eq": gpu_name}
        document = self._request("POST", "/api/v0/bundles/", json=query)
        return [sanitize_offer(offer) for offer in document.get("offers", [])]

    def create(self, offer: dict[str, Any], *, image: str, disk_gb: float, label: str) -> int:
        document = self._request(
            "PUT",
            f"/api/v0/asks/{int(offer['id'])}/",
            json={
                "image": image,
                "disk": disk_gb,
                "label": label,
                "runtype": "ssh_direct",
                "cancel_unavail": True,
                "env": {
                    "-p 9444:9444": "1",
                    "-p 29615:29615": "1",
                    "-p 9400:9400": "1",
                },
            },
        )
        instance_id = document.get("new_contract")
        if not isinstance(instance_id, int):
            raise ProviderError("Vast create response omitted the instance ID")
        return instance_id

    def register_ssh_key(self, public_key: str) -> dict[str, Any]:
        if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ")) or "PRIVATE" in public_key:
            raise ValueError("only an SSH public key may be registered")
        return self._request("POST", "/api/v0/ssh", json={"ssh_key": public_key})

    def instance(self, instance_id: int) -> dict[str, Any]:
        document = self._request("GET", f"/api/v0/instances/{instance_id}/")
        values = document.get("instances")
        if values is None:
            return {"id": instance_id, "actual_status": None}
        if not isinstance(values, dict):
            raise ProviderError("Vast instance response is malformed")
        return sanitize_instance(values)

    def instances(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        token = None
        for _ in range(100):
            parameters: dict[str, Any] = {"limit": 25}
            if token is not None:
                parameters["after_token"] = token
            try:
                document = self._request("GET", "/api/v0/instances/", params=parameters)
            except ProviderError as error:
                if "HTTP 410" in str(error):
                    return []
                raise
            results.extend(
                sanitize_instance(instance) for instance in document.get("instances", [])
            )
            token = document.get("next_token")
            if token is None:
                return results
        raise ProviderError("Vast instance pagination did not terminate")

    def wait_running(self, instance_id: int, timeout: float, guard: BudgetGuard) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            guard.require_below_limit()
            instance = self.instance(instance_id)
            status = instance.get("actual_status")
            if status == "running":
                return instance
            if status in TERMINAL_FAILURE_STATES:
                raise ProviderError(f"instance {instance_id} entered terminal state {status}")
            time.sleep(10)
        raise ProviderError(f"instance {instance_id} did not become ready before timeout")

    def charges(self, start: float, end: float) -> list[dict[str, Any]]:
        results = []
        token = None
        for _ in range(100):
            parameters: dict[str, Any] = {
                "select_filters": json.dumps(
                    {"day": {"gte": int(start) - 86400, "lte": int(end) + 86400}}
                ),
                "limit": 100,
            }
            if token is not None:
                parameters["after_token"] = token
            document = self._request("GET", "/api/v0/charges/", params=parameters)
            results.extend(document.get("results", []))
            token = document.get("next_token")
            if token is None:
                return results
        raise ProviderError("Vast charge pagination did not terminate")

    def refresh_charges(self, store: StateStore) -> dict[int, float]:
        snapshot = store.snapshot()
        instances = snapshot["instances"]
        if not instances:
            store.record_charges({})
            return {}
        start = min(float(record["created_at"]) for record in instances)
        charges = {record["id"]: 0.0 for record in instances}
        for charge in self.charges(start, time.time()):
            source = str(charge.get("source", ""))
            if not source.startswith("instance-"):
                continue
            try:
                instance_id = int(source.removeprefix("instance-"))
            except ValueError:
                continue
            if instance_id in charges:
                charges[instance_id] += float(charge.get("amount", 0.0))
        store.record_charges(charges)
        return charges

    def destroy(self, instance_id: int) -> None:
        try:
            self._request("DELETE", f"/api/v0/instances/{instance_id}/")
        except ProviderError as error:
            if "HTTP 404" not in str(error):
                raise

    def destroy_tracked(self, store: StateStore, guard: BudgetGuard) -> dict[str, Any]:
        errors = []
        tracked = store.snapshot()["instances"]
        active_ids = {record["id"] for record in tracked if "destroyed_at" not in record}
        for instance_id in sorted(active_ids):
            record = next(item for item in tracked if item["id"] == instance_id)
            estimate = guard.instance_estimate(record, time.time())
            for attempt in range(3):
                try:
                    self.destroy(instance_id)
                    store.mark_destroyed(instance_id, estimate)
                    break
                except ProviderError as error:
                    if attempt == 2:
                        errors.append({"id": instance_id, "error": type(error).__name__})
                    time.sleep(2**attempt)
        remaining = sorted(active_ids & {item["id"] for item in self.instances()})
        all_destroyed = not errors and not remaining
        report = {
            "schema_version": 1,
            "verified_at": time.time(),
            "tracked_instance_ids": sorted(record["id"] for record in tracked),
            "remaining_instance_ids": remaining,
            "errors": errors,
            "all_destroyed": all_destroyed,
            "unrelated_instances_untouched": True,
        }
        store.update("destruction_verified", all_destroyed=all_destroyed)
        if not all_destroyed:
            raise TeardownError(
                f"instances still require destruction: {remaining}; errors: {errors}"
            )
        return report


def mapped_tls_port(instance: dict[str, Any]) -> int:
    ports = instance.get("ports")
    bindings = ports.get("9444/tcp") if isinstance(ports, dict) else None
    if isinstance(bindings, list):
        for binding in bindings:
            value = binding.get("HostPort")
            if value is not None and 0 < int(value) <= 65535:
                return int(value)
    raise ProviderError(f"instance {instance.get('id')} has no external TLS port mapping")


def write_deployment_config(
    store: StateStore,
    instances: list[dict[str, Any]],
    *,
    model: str,
    checkpoint: Path,
    compiled: Path,
    output: Path,
    revision: str,
) -> dict[str, Any]:
    snapshot = store.snapshot()
    records = {record["id"]: record for record in snapshot["instances"]}
    workers = []
    for index, instance in enumerate(instances):
        record = records[int(instance["id"])]
        workers.append(
            {
                "name": f"worker-{chr(ord('a') + index)}",
                "host": instance["ssh_host"],
                "ssh_user": "root",
                "ssh_port": instance["ssh_port"],
                "tls_port": mapped_tls_port(instance),
                "public_host": instance.get("public_ipaddr") or instance["ssh_host"],
                "rendezvous_host": instances[0].get("public_ipaddr") or instances[0]["ssh_host"],
                "provider_instance_id": instance["id"],
                "machine_id": record["machine_id"],
                **({"ssh_key": record["ssh_key_path"]} if record.get("ssh_key_path") else {}),
            }
        )
    document = {
        "model": model,
        "checkpoint": str(checkpoint),
        "compiled": str(compiled),
        "output": str(output.parent / "deployment"),
        "revision": revision,
        "workers": workers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document
