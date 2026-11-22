"""The browser verifier's fixtures must be real output that the Python verifier accepts.

``apps/web-manual/tests/verify.test.ts`` checks the TypeScript verifier against this
file. Checking the same file here ties the two implementations to one set of facts:
what one accepts or rejects, the other must too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xurcroh.public_log import witnesses_of
from xurcroh.receipts import TraceError, trace_digest, verify_trace, verify_trace_proof

FIXTURE = Path(__file__).parents[2] / "apps/web-manual/tests/fixtures/verify.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_trace_verifies_and_opens_its_commitments(fixture: dict) -> None:
    signer = bytes.fromhex(fixture["signer_public"])
    report = verify_trace(
        fixture["trace"], public_key=signer, messages=fixture["messages"], output=fixture["output"]
    )
    assert report == {"signature": True, "prompt": True, "output": True, "worker_checkpoints": 2}
    # The pretty-printed text a user would hold parses back to the same signed document.
    assert json.loads(fixture["trace_text"]) == fixture["trace"]
    assert trace_digest(fixture["trace"]).hex() == fixture["trace_canonical_sha256"]
    assert verify_trace(fixture["media_trace"], media_input=fixture["media"])["prompt"] is True


def test_fixture_proof_and_witness_verify(fixture: dict) -> None:
    log_key = bytes.fromhex(fixture["log_key"])
    assert verify_trace_proof(fixture["trace"], fixture["proof"], log_key=log_key) == 7
    head = fixture["proof"]["tree_head"]
    assert len(witnesses_of(head, [bytes.fromhex(fixture["witness_key"])])) == 1
    assert witnesses_of(head, [bytes.fromhex(fixture["other_public"])]) == []


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("forged_checkpoint_trace", "checkpoint signature does not verify"),
        ("wrong_artifacts_trace", "signed for artifacts other"),
        ("wrong_manifest_trace", "manifest digest"),
    ],
)
def test_fixture_inner_forgeries_are_rejected(fixture: dict, name: str, reason: str) -> None:
    with pytest.raises(TraceError, match=reason):
        verify_trace(fixture[name])
