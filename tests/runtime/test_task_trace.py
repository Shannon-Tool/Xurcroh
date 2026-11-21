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

    assert trace["format"] == "xurcroh-task-trace-v1"
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
    assert report == {"signature": True, "prompt": True, "output": True}
    assert verify_trace(trace, public_key=public_key) == {
        "signature": True,
        "prompt": None,
        "output": None,
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
