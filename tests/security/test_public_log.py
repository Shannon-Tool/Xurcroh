from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xurcroh.public_log import (
    LogClient,
    PublicLog,
    PublicLogError,
    Witness,
    create_app,
    verify_entry_proof,
    witnesses_of,
)
from xurcroh.signatures import HybridSigningKey


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def serve(log: PublicLog, *, token: str | None = "write") -> LogClient:
    http = TestClient(create_app(log, write_token="write"))
    return LogClient("https://log.invalid", log.public_key, write_token=token, client=http)


def test_published_digest_gets_a_proof_anyone_can_check(tmp_path: Path) -> None:
    log = PublicLog(tmp_path, HybridSigningKey.generate())
    client = serve(log)

    for index in range(5):
        client.publish("trace", digest(f"other-{index}"))
    proof = client.publish("trace", digest("mine"))

    head = verify_entry_proof(proof, kind="trace", digest=digest("mine"), log_key=log.public_key)
    assert head["tree_size"] == 6
    assert client.find("trace", digest("mine"))["leaf_index"] == 5
    assert client.find("trace", digest("never logged")) is None
    # Publishing again does not create a second entry.
    assert client.publish("trace", digest("mine"))["leaf_index"] == 5
    assert log.size == 6


def test_proof_is_bound_to_digest_kind_and_log_key(tmp_path: Path) -> None:
    log = PublicLog(tmp_path, HybridSigningKey.generate())
    proof = serve(log).publish("trace", digest("mine"))

    with pytest.raises(PublicLogError, match="not in the log"):
        verify_entry_proof(proof, kind="trace", digest=digest("other"), log_key=log.public_key)
    with pytest.raises(PublicLogError, match="not in the log"):
        verify_entry_proof(proof, kind="manifest", digest=digest("mine"), log_key=log.public_key)
    other_key = HybridSigningKey.generate().public_key
    with pytest.raises(PublicLogError, match="not in the log"):
        verify_entry_proof(proof, kind="trace", digest=digest("mine"), log_key=other_key)
    moved = {**proof, "leaf_index": 1}
    with pytest.raises(PublicLogError, match="not in the log"):
        verify_entry_proof(moved, kind="trace", digest=digest("mine"), log_key=log.public_key)


def test_writing_needs_the_token_and_reading_does_not(tmp_path: Path) -> None:
    log = PublicLog(tmp_path, HybridSigningKey.generate())
    serve(log).publish("release", digest("build"))
    http = TestClient(create_app(log, write_token="write"))

    body = {"kind": "trace", "digest": digest("x")}
    assert http.post("/v1/entries", json=body).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert http.post("/v1/entries", json=body, headers=wrong).status_code == 401
    assert http.post("/v1/entries", json={"kind": "trace", "digest": "zz"}).status_code == 422
    assert http.get("/v1/tree-head").json()["tree_size"] == 1
    assert http.get("/v1/entries").json()["entries"] == [
        {"kind": "release", "digest": digest("build")}
    ]
    with pytest.raises(PublicLogError, match="write token"):
        serve(log, token=None).publish("trace", digest("x"))
    with pytest.raises(PublicLogError, match="HTTPS"):
        LogClient("http://log.invalid", log.public_key)
    with pytest.raises(ValueError, match="empty"):
        create_app(log, write_token="")


def test_log_survives_restart_with_the_same_history(tmp_path: Path) -> None:
    key = HybridSigningKey.generate()
    log = PublicLog(tmp_path, key)
    client = serve(log)
    proof = client.publish("manifest", digest("model"))
    client.publish("trace", digest("t"))

    restarted = PublicLog(tmp_path, key)

    assert restarted.size == 2
    assert restarted.tree_head()["root_hash"] == log.tree_head()["root_hash"]
    assert serve(restarted).extends(proof["tree_head"])["tree_size"] == 2
    assert (tmp_path / "entries.jsonl").stat().st_mode & 0o777 == 0o600


def test_a_rewritten_log_is_caught_by_anyone_holding_an_earlier_head(tmp_path: Path) -> None:
    key = HybridSigningKey.generate()
    honest = PublicLog(tmp_path / "honest", key)
    client = serve(honest)
    client.publish("trace", digest("a"))
    earlier = client.publish("trace", digest("b"))["tree_head"]
    client.publish("trace", digest("c"))
    assert client.extends(earlier)["tree_size"] == 3

    # The same operator key presents a history in which entry "b" never existed.
    rewritten = PublicLog(tmp_path / "rewritten", key)
    forged = serve(rewritten)
    for text in ("a", "x", "c"):
        forged.publish("trace", digest(text))
    with pytest.raises(PublicLogError, match="no longer contains the history"):
        forged.extends(earlier)
    # A log that simply dropped entries is caught the same way.
    truncated = PublicLog(tmp_path / "truncated", key)
    serve(truncated).publish("trace", digest("a"))
    with pytest.raises(PublicLogError, match="no longer contains the history"):
        serve(truncated).extends(earlier)


def test_witness_countersigns_only_heads_that_extend_what_it_saw(tmp_path: Path) -> None:
    key = HybridSigningKey.generate()
    log = PublicLog(tmp_path / "log", key)
    client = serve(log)
    client.publish("trace", digest("a"))
    witness = Witness(HybridSigningKey.generate(), tmp_path / "witness.json")

    witness.observe(client)
    client.publish("trace", digest("b"))
    witness.observe(client)

    head = client.find("trace", digest("b"))["tree_head"]
    assert witnesses_of(head, [witness.public_key]) != []
    assert witnesses_of(head, [HybridSigningKey.generate().public_key]) == []
    # Cosignatures survive a restart of the log.
    assert witnesses_of(PublicLog(tmp_path / "log", key).tree_head(), [witness.public_key]) != []

    rewritten = PublicLog(tmp_path / "rewritten", key)
    forged = serve(rewritten)
    for text in ("a", "x", "y"):
        forged.publish("trace", digest(text))
    with pytest.raises(PublicLogError, match="no longer contains the history"):
        witness.observe(forged)
    assert witnesses_of(rewritten.tree_head(), [witness.public_key]) == []


def test_forged_or_misplaced_cosignatures_are_refused(tmp_path: Path) -> None:
    log = PublicLog(tmp_path / "log", HybridSigningKey.generate())
    client = serve(log)
    client.publish("trace", digest("a"))
    witness = Witness(HybridSigningKey.generate(), tmp_path / "witness.json")
    cosignature = witness.observe(client)
    http = TestClient(create_app(log, write_token="write"))

    client.publish("trace", digest("b"))
    moved = {**cosignature, "tree_size": 2}
    assert http.post("/v1/cosignatures", json=moved).status_code == 400
    assert http.post("/v1/cosignatures", json={**cosignature, "tree_size": 9}).status_code == 400
    head = log.tree_head(1)
    tampered = {**head, "cosignatures": [{**head["cosignatures"][0], "signature": "00" * 3373}]}
    with pytest.raises(PublicLogError, match="witness signature"):
        witnesses_of(tampered, [witness.public_key])
    assert json.loads((tmp_path / "log" / "cosignatures.json").read_text()).keys() == {"1"}


def test_witness_refuses_an_empty_log(tmp_path: Path) -> None:
    log = PublicLog(tmp_path / "log", HybridSigningKey.generate())
    with pytest.raises(PublicLogError, match="nothing to witness"):
        Witness(HybridSigningKey.generate(), tmp_path / "w.json").observe(serve(log))


def test_an_unreachable_log_is_not_reported_as_a_rewritten_one(tmp_path: Path) -> None:
    log = PublicLog(tmp_path, HybridSigningKey.generate())
    head = serve(log).publish("trace", digest("a"))["tree_head"]

    class Down:
        def request(self, *_arguments: object, **_options: object) -> object:
            raise OSError("connection refused")

    offline = LogClient("https://log.invalid", log.public_key, client=Down())
    with pytest.raises(PublicLogError, match="request failed") as failure:
        offline.extends(head)
    assert "no longer contains" not in str(failure.value)
