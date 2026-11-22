from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError
from xurcroh.receipts import TraceError, verify_trace

MESSAGES = [{"role": "user", "content": "hello world"}]


def client_for(path: Path, **options: object) -> Xurcroh:
    return Xurcroh(cluster={"endpoint": "local://cpu"}, model_path={"tiny": path}, **options)


def test_trace_records_every_setting_and_verifies(tiny_model_path: Path) -> None:
    with client_for(tiny_model_path) as client:
        response = client.chat.completions.create(
            model="tiny",
            messages=MESSAGES,
            temperature=0.7,
            top_p=0.9,
            top_k=5,
            max_tokens=4,
            seed=3,
            stop=["zzz"],
        )
        trace = client.tasks.trace(response.execution.request_id)
        public_key = client.tasks.public_key
        digest = client.preflight("tiny")["tiny"].source_digest

    assert trace["format"] == "xurcroh-task-trace-v2"
    assert trace["signer"]["algorithm"] == "Ed25519+ML-DSA-65"
    assert trace["task_id"] == response.execution.request_id
    assert trace["mode"] == "local"
    assert trace["request"]["kind"] == "chat"
    assert trace["request"]["settings"] == {
        "max_tokens": 4,
        "model": "tiny",
        "seed": 3,
        "stop": ["zzz"],
        "stream": False,
        "temperature": 0.7,
        "timeout": None,
        "top_k": 5,
        "top_p": 0.9,
    }
    assert trace["request"]["effective"] == {
        "armoured_entities": None,
        "sampler": "xurcroh-multinomial-v2",
        "seed": 3,
    }
    assert trace["model"]["source_digest"] == digest
    assert trace["model"]["source_format"] == "safetensors"
    assert trace["model"]["engine"] == response.execution.engine
    assert trace["result"]["completion_tokens"] == response.usage.completion_tokens
    assert trace["security"]["mode"] == "strict"
    assert trace["security"]["attestation"] == "unattested"
    assert "hello world" not in json.dumps(trace)

    output = response.choices[0].message.content
    report = verify_trace(trace, public_key=public_key, messages=MESSAGES, output=output)
    # This model runs in-process, so no worker signed anything and the report says so.
    assert report == {
        "signature": True,
        "prompt": True,
        "output": True,
        "worker_checkpoints": None,
    }
    assert verify_trace(trace, public_key=public_key) == {
        "signature": True,
        "prompt": None,
        "output": None,
        "worker_checkpoints": None,
    }


def test_trace_verification_names_what_was_changed(tiny_model_path: Path) -> None:
    with client_for(tiny_model_path) as client:
        response = client.chat.completions.create(
            model="tiny", messages=MESSAGES, temperature=0, max_tokens=2
        )
        trace = client.tasks.trace(response.execution.request_id)
        public_key = client.tasks.public_key
    output = response.choices[0].message.content

    for path, value in (
        (("request", "settings", "temperature"), 1.0),
        (("model", "source_digest"), "0" * 64),
        (("model", "source_format"), "gguf"),
        (("security", "mode"), "tee"),
        (("result", "completion_tokens"), 99),
        (("task_id",), "other"),
    ):
        changed = copy.deepcopy(trace)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(TraceError, match="signature"):
            verify_trace(changed, public_key=public_key)
    with pytest.raises(TraceError, match="prompt"):
        verify_trace(trace, public_key=public_key, messages=[{"role": "user", "content": "other"}])
    with pytest.raises(TraceError, match="output"):
        verify_trace(trace, public_key=public_key, output=output + "x")
    with pytest.raises(TraceError, match="signature"):
        verify_trace(trace, public_key=bytes(32))
    unsigned = {key: value for key, value in trace.items() if key != "signature"}
    with pytest.raises(TraceError, match="signature"):
        verify_trace(unsigned, public_key=public_key)
    with pytest.raises(TraceError, match="format"):
        verify_trace({**trace, "format": "v0"}, public_key=public_key)


def test_trace_covers_streams_reports_drawn_seed_and_is_bounded(tiny_model_path: Path) -> None:
    with client_for(tiny_model_path, security={"trace_history": 2}) as client:
        first = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=2)
        chunks = list(
            client.chat.completions.create(
                model="tiny", messages=MESSAGES, max_tokens=2, stream=True
            )
        )
        execution = chunks[-1].execution
        assert execution is not None
        stream_trace = client.tasks.trace(execution.request_id)
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert verify_trace(
            stream_trace, public_key=client.tasks.public_key, messages=MESSAGES, output=text
        )["output"]
        assert stream_trace["request"]["settings"]["seed"] is None
        assert stream_trace["request"]["effective"]["seed"] == execution.seed
        assert stream_trace["request"]["settings"]["stream"] is True
        client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=2)
        with pytest.raises(ConfigurationError, match="no trace"):
            client.tasks.trace(first.execution.request_id)
        with pytest.raises(ConfigurationError, match="no trace"):
            client.tasks.trace("never-ran")


def test_verify_command_accepts_a_trace_and_rejects_a_changed_one(
    tiny_model_path: Path, tmp_path: Path
) -> None:
    with client_for(tiny_model_path) as client:
        response = client.chat.completions.create(
            model="tiny", messages=MESSAGES, temperature=0, max_tokens=2
        )
        trace = client.tasks.trace(response.execution.request_id)
        public_key = client.tasks.public_key.hex()
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(trace))
    trace["request"]["settings"]["max_tokens"] = 4000
    bad.write_text(json.dumps(trace))
    command = [sys.executable, "-m", "xurcroh.cli", "verify"]

    accepted = subprocess.run(  # noqa: S603
        [*command, str(good), "--public-key", public_key], capture_output=True, text=True
    )
    rejected = subprocess.run(  # noqa: S603
        [*command, str(bad), "--public-key", public_key], capture_output=True, text=True
    )

    assert accepted.returncode == 0, accepted.stderr
    assert "signature: verified" in accepted.stdout
    assert "temperature" in accepted.stdout
    assert rejected.returncode == 1
    assert "signature" in rejected.stderr


def test_trace_hash_is_anchored_in_an_append_only_log(tiny_model_path: Path) -> None:
    from xurcroh.receipts import verify_trace_proof

    with client_for(tiny_model_path) as client:
        ids = []
        for _ in range(3):
            response = client.chat.completions.create(
                model="tiny", messages=MESSAGES, temperature=0, max_tokens=1
            )
            ids.append(response.execution.request_id)
        traces = [client.tasks.trace(task_id) for task_id in ids]
        proofs = [client.tasks.proof(task_id) for task_id in ids]
        log_key = client.tasks.log_key

    assert [proof["leaf_index"] for proof in proofs] == [0, 1, 2]
    assert "hello world" not in json.dumps(proofs)
    for trace, proof in zip(traces, proofs, strict=True):
        assert verify_trace_proof(trace, proof, log_key=log_key) == proof["tree_head"]["tree_size"]
    rewritten = copy.deepcopy(traces[0])
    rewritten["result"]["completion_tokens"] = 7
    with pytest.raises(TraceError, match="log"):
        verify_trace_proof(rewritten, proofs[0], log_key=log_key)
    with pytest.raises(TraceError, match="log"):
        verify_trace_proof(traces[0], proofs[1], log_key=log_key)
    with pytest.raises(TraceError, match="log"):
        verify_trace_proof(traces[0], proofs[0], log_key=bytes(32))


def test_trace_binds_the_model_to_one_manifest_digest(tiny_model_path: Path) -> None:
    from xurcroh.receipts import model_manifest_digest, sign_trace
    from xurcroh.signatures import HybridSigningKey

    with client_for(tiny_model_path) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        trace = client.tasks.trace(response.execution.request_id)

    model = trace["model"]
    assert model["repository"] is None  # a local checkpoint has no repository name
    assert model["manifest_sha256"] == model_manifest_digest(model)
    for name in ("revision", "source_digest", "source_format", "share_count", "weight_bits"):
        assert model_manifest_digest({**model, name: "changed"}) != model["manifest_sha256"], name
    # Even a trace re-signed by someone else cannot pair the digest with other model facts.
    forged = sign_trace(
        {**trace, "model": {**model, "source_format": "gguf"}}, HybridSigningKey.generate()
    )
    with pytest.raises(TraceError, match="manifest"):
        verify_trace(forged)


def test_a_classical_only_signature_is_not_accepted(tiny_model_path: Path) -> None:
    """Dropping the post-quantum half must not leave a trace that still verifies."""

    with client_for(tiny_model_path) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        trace = client.tasks.trace(response.execution.request_id)

    classical_only = {
        **trace,
        "signer": {"algorithm": "Ed25519", "public_key": trace["signer"]["public_key"][:64]},
        "signature": trace["signature"][:128],
    }
    with pytest.raises(TraceError, match="signature"):
        verify_trace(classical_only)
    with pytest.raises(TraceError, match="signature"):
        verify_trace({**trace, "signature": trace["signature"][:128]})


def _public_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A live log service the client under test publishes to."""

    from fastapi.testclient import TestClient

    from xurcroh.public_log import LogClient, PublicLog, create_app
    from xurcroh.runtime import service
    from xurcroh.signatures import HybridSigningKey

    log = PublicLog(tmp_path / "log", HybridSigningKey.generate())
    http = TestClient(create_app(log, write_token="write"))
    client = LogClient("https://log.invalid", log.public_key, write_token="write", client=http)
    monkeypatch.setattr(service, "public_log_client", lambda security: client)
    settings = {"trace_log_endpoint": "https://log.invalid", "trace_log_key": log.public_key.hex()}
    return log, client, settings


def test_trace_hash_is_published_to_a_public_log_and_checked_from_outside(
    tiny_model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.receipts import trace_digest, verify_trace_proof

    log, log_client, settings = _public_log(tmp_path, monkeypatch)
    with client_for(tiny_model_path, security=settings) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        trace = client.tasks.trace(response.execution.request_id)
        proof = client.tasks.proof(response.execution.request_id)
        assert client.tasks.log_key == log.public_key

    # The log holds only a hash of the trace, and anyone with the log key can check it.
    assert log.entries(0, 1) == [{"kind": "trace", "digest": trace_digest(trace).hex()}]
    assert verify_trace_proof(trace, proof, log_key=log.public_key) == 1
    assert trace["security"]["public_log"]["endpoint"] == "https://log.invalid"
    assert log_client.extends(proof["tree_head"])["tree_size"] == 1
    changed = copy.deepcopy(trace)
    changed["result"]["completion_tokens"] += 1
    with pytest.raises(TraceError, match="not in the log"):
        verify_trace_proof(changed, proof, log_key=log.public_key)


def test_log_outage_keeps_the_trace_and_the_proof_is_retried(
    tiny_model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.public_log import PublicLogError

    log, log_client, settings = _public_log(tmp_path, monkeypatch)
    working = log_client._client

    class Down:
        def request(self, *_arguments: object, **_options: object) -> object:
            raise OSError("connection refused")

    with client_for(tiny_model_path, security=settings) as client:
        log_client._client = Down()
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        task = response.execution.request_id
        assert client.tasks.trace(task)["task_id"] == task
        with pytest.raises(PublicLogError, match="request failed"):
            client.tasks.proof(task)
        log_client._client = working
        assert client.tasks.proof(task)["leaf_index"] == 0
    assert log.size == 1


def test_model_whose_manifest_is_not_logged_is_refused_when_required(
    tiny_model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log, log_client, settings = _public_log(tmp_path, monkeypatch)
    required = {**settings, "require_logged_manifest": True}

    with client_for(tiny_model_path, security=settings) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        manifest = client.tasks.trace(response.execution.request_id)["model"]["manifest_sha256"]
    with (
        client_for(tiny_model_path, security=required) as client,
        pytest.raises(ConfigurationError, match="not in the public log"),
    ):
        client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)

    log_client.publish("manifest", manifest)
    with client_for(tiny_model_path, security=required) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        trace = client.tasks.trace(response.execution.request_id)
    assert trace["security"]["public_log"]["manifest_required"] is True
    assert log.lookup("manifest", manifest) is not None


def test_client_whose_own_build_is_not_logged_is_refused_when_required(
    tiny_model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xurcroh.build import installed_build_digest

    log, log_client, settings = _public_log(tmp_path, monkeypatch)
    required = {**settings, "require_logged_build": True}

    # Refused at start, before any model is loaded or any request is served.
    with pytest.raises(ConfigurationError, match="client build .* is not in the public log"):
        client_for(tiny_model_path, security=required)

    log_client.publish("build", installed_build_digest())
    with client_for(tiny_model_path, security=required) as client:
        response = client.chat.completions.create(model="tiny", messages=MESSAGES, max_tokens=1)
        trace = client.tasks.trace(response.execution.request_id)
    assert trace["client"]["build_sha256"] == installed_build_digest()
    assert trace["security"]["public_log"]["build_required"] is True


def test_public_log_settings_must_be_complete() -> None:
    from xurcroh.config import SecurityConfig

    with pytest.raises(ValueError, match="set together"):
        SecurityConfig(trace_log_endpoint="https://log.invalid")
    with pytest.raises(ValueError, match="needs a trace_log_endpoint"):
        SecurityConfig(require_logged_manifest=True)
    with pytest.raises(ValueError, match="needs a trace_log_endpoint"):
        SecurityConfig(require_logged_build=True)
