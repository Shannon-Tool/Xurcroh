from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import httpx
import pytest

from xurcroh.errors import BudgetExceededError, ProviderError, TeardownError
from xurcroh.providers.vast import (
    BudgetGuard,
    BudgetPolicy,
    StateStore,
    VastClient,
    sanitize_offer,
    write_deployment_config,
)


def offer(instance_id: int = 10, machine_id: int = 20) -> dict:
    return {
        "id": instance_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24576,
        "num_gpus": 1,
        "dph_total": 0.4,
        "storage_total_cost": 0.002,
        "reliability": 0.999,
    }


def record(now: float) -> dict:
    return {
        "id": 123,
        "offer_id": 10,
        "machine_id": 20,
        "created_at": now - 3600,
        "compute_hourly_usd": 0.4,
        "storage_hourly_usd": 0.002,
        "bandwidth_usd": 0.1,
    }


def client_for(handler) -> VastClient:
    client = httpx.Client(
        base_url="https://console.vast.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test-credential"},
    )
    return VastClient("test-credential", client=client)


def test_budget_guard_includes_compute_storage_bandwidth_and_reported_charges(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "state.json")
    now = time.time()
    store.add_instance(record(now))
    guard = BudgetGuard(store)
    assert guard.total(now=now) == pytest.approx(0.502)
    assert guard.forecast(offer(), 2, 40) == pytest.approx(0.904)
    store.record_charges({123: 1.25})
    assert guard.total(now=now) == pytest.approx(1.25)
    with pytest.raises(BudgetExceededError, match="exceeds"):
        guard.require_capacity(24)
    with pytest.raises(ValueError, match="finite"):
        guard.require_capacity(float("nan"))
    with pytest.raises(ValueError, match="hard limit"):
        BudgetPolicy(target_usd=31)


def test_uncapped_budget_retains_accounting_without_a_spend_threshold(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    now = time.time()
    store.add_instance(record(now))
    guard = BudgetGuard(store, BudgetPolicy.uncapped())
    guard.require_capacity(1_000_000)
    guard.require_below_limit()
    assert guard.total(now=now) == pytest.approx(0.502)
    with pytest.raises(ValueError, match="finite"):
        guard.require_capacity(float("inf"))


def test_state_store_is_atomic_and_does_not_contain_api_keys(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.add_instance(record(time.time()))
    store.mark_destroyed(123, 0.5)
    store.record_charges({123: 0.6})
    snapshot = store.snapshot()
    assert snapshot["instances"][0]["estimated_final_usd"] == 0.5
    assert snapshot["instances"][0]["reported_charges_usd"] == 0.6
    assert "test-credential" not in path.read_text()
    with pytest.raises(ProviderError, match="already tracked"):
        store.add_instance(record(time.time()))


def test_offer_sanitizer_preserves_operator_and_capacity_metadata() -> None:
    result = sanitize_offer(
        {
            **offer(),
            "host_id": 483833,
            "cpu_name": "AMD EPYC 7542 32-Core Processor",
            "cpu_ram": 36822,
            "driver_version": "570.195.03",
            "secret": "excluded",
        }
    )
    assert result["provider_id"] == "vast-host-483833"
    assert result["cpu_vendor"] == "AMD"
    assert result["cpu_ram"] == 36822
    assert result["driver_version"] == "570.195.03"
    assert "secret" not in result


def test_capacity_eligibility_excludes_selected_driver_series() -> None:
    from scripts.vast.provision import eligible_offers

    args = type(
        "Arguments",
        (),
        {
            "min_gpu_ram_gb": 1,
            "min_host_ram_gb": 0,
            "max_hourly_usd": 10,
            "driver_version": None,
            "exclude_driver_prefix": ["580"],
        },
    )()
    compatible = {**offer(), "driver_version": "595.91.07"}
    incompatible = {**offer(11, 21), "driver_version": "580.159.03"}
    assert eligible_offers(args, [incompatible, compatible]) == [compatible]


def test_offers_create_and_instance_responses_are_sanitized() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v0/bundles/":
            return httpx.Response(
                200, json={"success": True, "offers": [{**offer(), "secret": "x"}]}
            )
        if request.url.path == "/api/v0/asks/10/":
            body = json.loads(request.content)
            assert "VAST_API_KEY" not in str(body)
            assert body["runtype"] == "ssh_direct"
            assert "-p 9400:9400" in body["env"]
            return httpx.Response(200, json={"new_contract": 123, "instance_api_key": "private"})
        if request.url.path == "/api/v0/instances/123/":
            return httpx.Response(
                200,
                json={
                    "instances": {
                        "id": 123,
                        "actual_status": "running",
                        "status_msg": "image download in progress",
                        "ssh_host": "ssh.example",
                        "ssh_port": 2200,
                        "ports": {"9444/tcp": [{"HostPort": "54304"}]},
                        "jupyter_token": "private",
                    }
                },
            )
        return httpx.Response(404)

    client = client_for(handler)
    offers = client.offers("RTX 4090")
    assert "secret" not in offers[0]
    assert client.create(offers[0], image="vllm:test", disk_gb=40, label="validation") == 123
    instance = client.instance(123)
    assert instance["ssh_host"] == "ssh.example"
    assert instance["status_msg"] == "image download in progress"
    assert instance["ports"]["9444/tcp"][0]["HostPort"] == "54304"
    assert "jupyter_token" not in instance
    query = json.loads(requests[0].content)
    assert query["verified"] == {"eq": True}
    assert query["num_gpus"] == {"eq": 1}
    assert query["reliability"] == {"gte": 0.99}


def test_wait_running_handles_terminal_states_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    statuses = iter(["loading", "running"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": {"id": 123, "actual_status": next(statuses)}})

    guard = BudgetGuard(StateStore(tmp_path / "state.json"))
    client = client_for(handler)
    assert client.wait_running(123, 1, guard)["actual_status"] == "running"

    failed = client_for(
        lambda request: httpx.Response(
            200, json={"instances": {"id": 123, "actual_status": "exited"}}
        )
    )
    with pytest.raises(ProviderError, match="terminal state"):
        failed.wait_running(123, 1, guard)


def test_charge_reconciliation_ignores_unrelated_instances(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.add_instance(record(time.time()))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/charges/"
        assert "day" in request.url.params["select_filters"]
        return httpx.Response(
            200,
            json={
                "success": True,
                "results": [
                    {"source": "instance-123", "amount": 0.75},
                    {"source": "instance-999", "amount": 100},
                    {"source": "volume-123", "amount": 50},
                ],
                "next_token": None,
            },
        )

    client = client_for(handler)
    assert client.refresh_charges(store) == {123: 0.75}
    assert store.snapshot()["instances"][0]["reported_charges_usd"] == 0.75


def test_destroy_tracked_verifies_remote_absence_and_preserves_unrelated(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.add_instance(record(time.time()))
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            deleted.append(123)
            return httpx.Response(200, json={"success": True})
        return httpx.Response(
            200,
            json={
                "success": True,
                "instances": [{"id": 999, "actual_status": "running"}],
                "next_token": None,
            },
        )

    client = client_for(handler)
    guard = BudgetGuard(store)
    report = client.destroy_tracked(store, guard)
    assert deleted == [123]
    assert report["all_destroyed"] is True
    assert report["remaining_instance_ids"] == []
    assert report["unrelated_instances_untouched"] is True
    assert "destroyed_at" in store.snapshot()["instances"][0]
    assert client.destroy_tracked(store, guard)["tracked_instance_ids"] == [123]


def test_destroy_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StateStore(tmp_path / "state.json")
    store.add_instance(record(time.time()))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(500)
        return httpx.Response(200, json={"instances": [{"id": 123}], "next_token": None})

    client = client_for(handler)
    with pytest.raises(TeardownError, match="still require destruction"):
        client.destroy_tracked(store, BudgetGuard(store))


def test_deployment_config_has_distinct_hosts_and_no_credentials(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.add_instance({**record(time.time()), "id": 123, "machine_id": 20})
    store.add_instance({**record(time.time()), "id": 456, "machine_id": 21})
    instances = [
        {
            "id": 123,
            "ssh_host": "ssh-a",
            "ssh_port": 22,
            "public_ipaddr": "192.0.2.1",
            "ports": {"9444/tcp": [{"HostPort": "54304"}]},
        },
        {
            "id": 456,
            "ssh_host": "ssh-b",
            "ssh_port": 23,
            "public_ipaddr": "192.0.2.2",
            "ports": {"9444/tcp": [{"HostPort": "54986"}]},
        },
    ]
    output = tmp_path / "two-node.json"
    config = write_deployment_config(
        store,
        instances,
        model="tiny",
        checkpoint=tmp_path / "model",
        compiled=tmp_path / "compiled",
        output=output,
        revision="abc",
    )
    assert config["output"] == str(tmp_path / "deployment")
    assert config["workers"][0]["machine_id"] != config["workers"][1]["machine_id"]
    assert config["workers"][0]["tls_port"] == 54304
    assert config["workers"][1]["tls_port"] == 54986
    assert config["workers"][1]["rendezvous_host"] == "192.0.2.1"
    assert "credential" not in output.read_text()


def test_watchdog_destroys_after_budget_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = Path(__file__).resolve().parents[2] / "scripts/vast/provision.py"
    spec = importlib.util.spec_from_file_location("vast_provision", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = StateStore(tmp_path / "state.json")
    store.add_instance(record(time.time()))
    destroyed = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def refresh_charges(self, store):
            store.record_charges({123: 26.0})

        def destroy_tracked(self, store, guard):
            destroyed.append(123)
            store.mark_destroyed(123, 26.0)
            return {"all_destroyed": True}

    monkeypatch.setattr(module, "VastClient", FakeClient)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    module.watchdog(tmp_path / "state.json", 3600, BudgetPolicy())
    assert destroyed == [123]


def test_uncapped_provision_policy_is_explicit() -> None:
    module_path = Path(__file__).resolve().parents[2] / "scripts/vast/provision.py"
    spec = importlib.util.spec_from_file_location("vast_provision_policy", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = type("Arguments", (), {"uncapped": True, "target_usd": 25, "hard_limit_usd": 30})()
    policy = module.budget_policy(args)
    assert policy.target_usd is None
    assert policy.hard_limit_usd is None


def test_null_instance_is_accepted_while_provisioning() -> None:
    client = client_for(lambda request: httpx.Response(200, json={"instances": None}))
    assert client.instance(123) == {"id": 123, "actual_status": None}


def test_deleted_instance_collection_is_empty() -> None:
    client = client_for(lambda request: httpx.Response(410))
    assert client.instances() == []


def test_rate_limited_request_uses_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json={"success": True, "offers": []})

    monkeypatch.setattr(time, "sleep", lambda seconds: delays.append(seconds))
    client = client_for(handler)
    assert client.offers() == []
    assert len(attempts) == 2
    assert delays == [2.0]


def test_diverse_offer_selection_supports_four_hosts_and_cost_guards(tmp_path: Path) -> None:
    from xurcroh.providers.vast import select_diverse_offers

    offers = [
        {
            "id": index,
            "machine_id": index,
            "gpu_ram": 24_576,
            "dph_total": 0.2,
            "geolocation": "de" if index % 2 else "us",
            "cpu_vendor": "amd" if index % 3 else "intel",
            "provider_id": "vast" if index % 2 else "other",
        }
        for index in range(6)
    ]
    store = StateStore(tmp_path / "state.json")
    guard = BudgetGuard(store)
    selected, forecast = select_diverse_offers(
        offers, count=4, guard=guard, planned_hours=1, disk_gb=40
    )
    assert len(selected) == 4
    assert len({offer["machine_id"] for offer in selected}) == 4
    assert len({offer["geolocation"] for offer in selected}) == 2
    assert len({offer["cpu_vendor"] for offer in selected}) == 2
    assert len({offer["provider_id"] for offer in selected}) == 2
    assert forecast == pytest.approx(1.2)
    with pytest.raises(ProviderError, match="diversity"):
        select_diverse_offers(
            [{**offer, "geolocation": "us", "cpu_vendor": "amd"} for offer in offers],
            count=4,
            guard=guard,
            planned_hours=1,
            disk_gb=40,
        )


def test_multi_host_selection_defers_unknown_cpu_vendor_to_hardware(tmp_path: Path) -> None:
    from xurcroh.providers.vast import select_diverse_offers

    offers = [
        {
            **offer(index + 1, index + 10),
            "geolocation": f"country-{index}",
            "provider_id": f"operator-{index}",
            "cpu_vendor": "unknown" if index == 0 else "amd",
        }
        for index in range(4)
    ]
    selected, _ = select_diverse_offers(
        offers,
        count=4,
        guard=BudgetGuard(StateStore(tmp_path / "state.json")),
        planned_hours=1,
        disk_gb=40,
        max_country_fraction=1,
        max_cpu_vendor_fraction=1,
        max_provider_fraction=1,
    )
    assert selected[0]["cpu_vendor"] == "unknown"


def test_provision_preserves_offer_diversity_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = Path(__file__).resolve().parents[2] / "scripts/vast/provision.py"
    spec = importlib.util.spec_from_file_location("vast_provision_metadata", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected_offer = {
        "id": 10,
        "machine_id": 20,
        "gpu_ram": 24_576,
        "dph_total": 0.4,
        "storage_total_cost": 0.002,
        "provider_id": "provider-a",
        "geolocation": "de",
        "cpu_vendor": "amd",
    }

    class FakeClient:
        def __init__(self):
            self.store = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def refresh_charges(self, store):
            return {}

        def offers(self, gpu_name, limit=0):
            FakeClient.offer_limit = limit
            return [selected_offer]

        def create(self, offer, **kwargs):
            self.create_attempts = getattr(self, "create_attempts", 0) + 1
            if self.create_attempts == 1:
                raise ProviderError("Vast API returned HTTP 400")
            return 123

        def destroy_tracked(self, store, guard):
            return {"all_destroyed": True}

        def wait_running(self, instance_id, timeout, guard):
            return {
                "id": 123,
                "ssh_host": "192.0.2.1",
                "ssh_port": 22,
                "public_ipaddr": "192.0.2.1",
                "ports": {"9444/tcp": [{"HostPort": "9444"}]},
            }

    class Process:
        pid = 1
        returncode = 0

        def __init__(self, *args):
            self.args = args[0] if args else []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def communicate(self, input=None, timeout=None):
            return ("abc", "")

        def poll(self):
            return 0

    monkeypatch.setattr(module, "VastClient", FakeClient)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    args = type("Arguments", (), {})()
    args.__dict__.update(
        {
            "state": tmp_path / "state.json",
            "count": 1,
            "uncapped": True,
            "target_usd": 25,
            "hard_limit_usd": 30,
            "gpu_name": None,
            "min_gpu_ram_gb": 1,
            "min_host_ram_gb": 0,
            "offer_limit": 200,
            "max_hourly_usd": 10,
            "planned_hours": 1,
            "disk_gb": 40,
            "max_country_fraction": 1,
            "max_cpu_vendor_fraction": 1,
            "max_provider_fraction": 1,
            "boot_timeout": 1,
            "ttl_seconds": 60,
            "image": "test-image",
            "model": "tiny",
            "checkpoint": tmp_path / "model",
            "compiled": tmp_path / "compiled",
            "output": tmp_path / "output",
        }
    )
    module.provision(args)
    stored = module.StateStore(tmp_path / "state.json").snapshot()["instances"][0]
    assert stored["provider_id"] == "provider-a"
    assert stored["geolocation"] == "de"
    assert stored["cpu_vendor"] == "amd"
    assert module.VastClient.offer_limit == 200


def test_register_ssh_key_uses_only_public_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v0/ssh"
        body = json.loads(request.content)
        assert body["ssh_key"].startswith("ssh-ed25519 ")
        assert "PRIVATE" not in body["ssh_key"]
        return httpx.Response(200, json={"success": True, "key": {"id": 1}})

    client = client_for(handler)
    result = client.register_ssh_key("ssh-ed25519 public-test-key")
    assert result["success"] is True
