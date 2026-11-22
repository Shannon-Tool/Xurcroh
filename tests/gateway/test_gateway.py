from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError
from xurcroh.gateway import create_gateway_app, key_digest, load_api_keys
from xurcroh.receipts import TraceError, verify_trace
from xurcroh.signatures import HybridSigningKey

MESSAGES = [{"role": "user", "content": "hello world"}]
ALICE = {"Authorization": "Bearer alice-key"}
BOB = {"Authorization": "Bearer bob-key"}
KEYS = {key_digest("alice-key"): "alice", key_digest("bob-key"): "bob"}


@pytest.fixture
def gateway(tiny_model_path: Path):
    signing_key = HybridSigningKey.generate()
    client = Xurcroh(cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path})
    app = create_gateway_app(client, api_keys=KEYS, signing_key=signing_key)
    with TestClient(app) as http:
        yield http, signing_key


def _chat(http: TestClient, headers: dict[str, str], **options: object):
    body = {"model": "tiny", "messages": MESSAGES, "max_tokens": 2, "temperature": 0, **options}
    return http.post("/v1/chat/completions", json=body, headers=headers)


def test_requester_gets_a_task_id_and_a_trace_marked_operator_asserted(gateway) -> None:
    http, signing_key = gateway

    response = _chat(http, ALICE, seed=5)
    task = response.headers["X-Xurcroh-Task-Id"]
    trace = http.get(f"/v1/tasks/{task}/trace", headers=ALICE).json()
    published = http.get("/v1/trace-key").json()

    assert response.status_code == 200
    assert response.json()["execution"]["request_id"] == task
    assert trace["task_id"] == task
    assert trace["mode"] == "hosted"
    assert trace["assurance"] == "operator-asserted"
    assert trace["request"]["settings"]["seed"] == 5
    assert published["public_key"] == signing_key.public_key.hex()
    assert published["assurance"] == "operator-asserted"
    output = response.json()["choices"][0]["message"]["content"]
    report = verify_trace(
        trace, public_key=signing_key.public_key, messages=MESSAGES, output=output
    )
    assert report["signature"] and report["prompt"] and report["output"]
    assert "hello world" not in json.dumps(trace)
    # The label is inside the signature, so it cannot be stripped to pass as a local trace.
    with pytest.raises(TraceError, match="signature"):
        verify_trace({**trace, "mode": "local"}, public_key=signing_key.public_key)
    unlabelled = {name: value for name, value in trace.items() if name != "assurance"}
    with pytest.raises(TraceError, match="signature"):
        verify_trace(unlabelled, public_key=signing_key.public_key)


def test_only_the_requester_can_read_a_task(gateway) -> None:
    http, _ = gateway
    task = _chat(http, ALICE).headers["X-Xurcroh-Task-Id"]

    assert http.get(f"/v1/tasks/{task}/trace", headers=ALICE).status_code == 200
    assert http.get(f"/v1/tasks/{task}/proof", headers=ALICE).status_code == 200
    other = http.get(f"/v1/tasks/{task}/trace", headers=BOB)
    missing = http.get("/v1/tasks/task-does-not-exist/trace", headers=BOB)
    assert other.status_code == missing.status_code == 404
    assert other.json() == missing.json()  # indistinguishable, so IDs cannot be probed
    assert http.get(f"/v1/tasks/{task}/proof", headers=BOB).status_code == 404
    assert http.get(f"/v1/tasks/{task}/trace").status_code == 401


def test_requests_need_a_known_api_key(gateway) -> None:
    http, _ = gateway

    assert _chat(http, {}).status_code == 401
    assert _chat(http, {"Authorization": "Bearer wrong"}).status_code == 401
    assert _chat(http, {"Authorization": "alice-key"}).status_code == 401
    assert http.get("/health").status_code == 200


def test_requester_cannot_choose_or_reuse_a_task_id(gateway) -> None:
    http, _ = gateway
    first = _chat(http, ALICE, request_id="mine").headers["X-Xurcroh-Task-Id"]
    second = _chat(http, BOB, request_id=first).headers["X-Xurcroh-Task-Id"]

    assert first.startswith("task-") and second.startswith("task-") and first != second
    assert http.get(f"/v1/tasks/{first}/trace", headers=ALICE).json()["task_id"] == first
    assert http.get(f"/v1/tasks/{first}/trace", headers=BOB).status_code == 404


def test_streamed_request_has_a_task_id_and_a_trace(gateway) -> None:
    http, signing_key = gateway

    response = _chat(http, ALICE, stream=True)
    task = response.headers["X-Xurcroh-Task-Id"]
    trace = http.get(f"/v1/tasks/{task}/trace", headers=ALICE).json()

    assert response.status_code == 200 and "data: [DONE]" in response.text
    assert trace["request"]["settings"]["stream"] is True
    assert verify_trace(trace, public_key=signing_key.public_key)["signature"]


def test_errors_are_reported_without_a_server_fault(gateway) -> None:
    http, _ = gateway

    unknown = http.post(
        "/v1/chat/completions", json={"model": "nope", "messages": MESSAGES}, headers=ALICE
    )
    assert unknown.status_code in {400, 404}
    assert unknown.json()["error"]["message"]


def test_api_key_file_holds_digests_only() -> None:
    text = f"# comment\nalice:{key_digest('alice-key')}\n\nbob:{key_digest('bob-key')}\n"

    assert load_api_keys(text) == KEYS
    for bad in (
        "",
        "alice",
        "alice:short",
        f"a:{key_digest('k')}\nb:{key_digest('k')}",
        "a:" + "z" * 64,
    ):
        with pytest.raises(ConfigurationError):
            load_api_keys(bad)


def test_trace_settings_are_fixed_once_a_trace_exists(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        with pytest.raises(ConfigurationError, match="cannot change"):
            create_gateway_app(client, api_keys=KEYS)
        with pytest.raises(ConfigurationError, match="must state its assurance"):
            client._runtime.configure_traces(mode="hosted")


def test_verify_command_says_a_hosted_trace_is_the_operators_statement(
    gateway, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from xurcroh import cli

    http, signing_key = gateway
    task = _chat(http, ALICE).headers["X-Xurcroh-Task-Id"]
    (tmp_path / "trace.json").write_text(http.get(f"/v1/tasks/{task}/trace", headers=ALICE).text)
    (tmp_path / "proof.json").write_text(http.get(f"/v1/tasks/{task}/proof", headers=ALICE).text)
    arguments = argparse.Namespace(
        trace=tmp_path / "trace.json",
        public_key=signing_key.public_key.hex(),
        messages=None,
        media_input=None,
        output=None,
        proof=tmp_path / "proof.json",
        log_key=http.get("/v1/trace-key").json()["log_key"],
        log_url=None,
        witness_key=[],
    )

    assert cli._verify(arguments) == 0
    output = capsys.readouterr().out
    assert "assurance: operator-asserted" in output
    assert "operator's statement" in output
    assert "log: included at tree size 1" in output
