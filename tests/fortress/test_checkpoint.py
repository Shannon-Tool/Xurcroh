from __future__ import annotations

import pytest
import torch
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.errors import ProtocolError
from xurcroh.fortress.arithmetic import split_secret
from xurcroh.fortress.checkpoint import (
    CHECKPOINT_OPERATION,
    SessionTranscript,
    artifact_set_digest,
    verify_checkpoint_signature,
)
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.transport import (
    FortressGridRuntime,
    MutualTLSIdentity,
    RoleEndpoint,
    encode_role_artifact,
)
from xurcroh.fortress.worker import FortressRoleWorker
from xurcroh.kem import HybridKemKey
from xurcroh.signatures import HybridSigningKey


def test_artifact_set_digest_ignores_order_and_detects_any_change() -> None:
    digest = artifact_set_digest({"a": b"one", "b": b"two"})

    assert digest == artifact_set_digest({"b": b"two", "a": b"one"})
    assert digest != artifact_set_digest({"a": b"one", "b": b"twO"})
    assert digest != artifact_set_digest({"a": b"one"})
    assert digest != artifact_set_digest({"a": b"two", "b": b"one"})


def test_transcript_depends_on_session_order_and_boundaries() -> None:
    first = SessionTranscript("s")
    first.update(b"ab", b"c")
    second = SessionTranscript("s")
    second.update(b"a", b"bc")
    other_session = SessionTranscript("t")
    other_session.update(b"ab", b"c")

    assert first.calls == 1
    assert len({first.hexdigest(), second.hexdigest(), other_session.hexdigest()}) == 3


class _Grid:
    """Four in-process role workers with identity keys, and a client runtime."""

    def __init__(
        self,
        mtls: MutualTLSIdentities,
        *,
        pin_artifacts: bool = True,
        swap_identity: bool = False,
        swap_artifacts: bool = False,
        signing: bool = True,
    ) -> None:
        self.operation = compile_operation(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
        )
        identity = MutualTLSIdentity(
            certificates=(mtls["client_certificate"], mtls["ca"]),
            private_key=mtls["client_key"],
            pinned_ca=mtls["ca"],
        )
        self.workers: list[FortressRoleWorker] = []
        roles = []
        for activation_index in range(2):
            for weight_index in range(2):
                private = HybridKemKey.generate()
                public = private.public_key
                artifacts = {"op": encode_role_artifact(self.operation, weight_index, public)}
                key = HybridSigningKey.generate()
                worker = FortressRoleWorker(
                    artifacts,
                    private.to_bytes(),
                    certificates=(mtls["server_certificate"], mtls["ca"]),
                    private_key=mtls["server_key"],
                    client_ca=mtls["ca"],
                    allow_unattested=True,
                    identity_key=key.to_bytes() if signing else None,
                )
                worker.start()
                self.workers.append(worker)
                pinned = HybridSigningKey.generate() if swap_identity else key
                expected = artifact_set_digest(
                    {"op": b"something else"} if swap_artifacts else artifacts
                )
                roles.append(
                    RoleEndpoint(
                        activation_index,
                        weight_index,
                        worker.address,
                        public,
                        f"role-{activation_index}-{weight_index}",
                        identity_public=pinned.public_key,
                        artifact_digest=expected if pin_artifacts else None,
                    )
                )
        self.runtime = FortressGridRuntime(roles, identity, cover_interval=60)
        self.client = self.runtime.client(self.operation, "op")

    def call(self) -> None:
        self.client.execute(split_secret(torch.tensor([5, 7]), count=2).shares)

    def close(self) -> None:
        self.runtime.close()
        for worker in self.workers:
            worker.close()


def test_worker_reports_the_digest_of_the_artifacts_it_loaded(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        assert all(len(worker.artifact_digest) == 64 for worker in grid.workers)
    finally:
        grid.close()


def test_every_role_returns_a_signed_checkpoint_matching_the_client_transcript(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        grid.call()
        grid.call()
        checkpoints = grid.runtime.checkpoints()

        assert [item["worker"] for item in checkpoints] == [
            "role-0-0",
            "role-0-1",
            "role-1-0",
            "role-1-1",
        ]
        for item, worker in zip(checkpoints, grid.workers, strict=True):
            body = verify_checkpoint_signature(item)
            assert body["calls"] == 2
            assert body["artifacts"] == worker.artifact_digest
            assert item["artifacts_pinned"] is True
        # The checkpoint exchange is itself part of the session, so the next one moves on.
        grid.call()
        later = grid.runtime.checkpoints()
        assert [item["body"]["calls"] for item in later] == [4, 4, 4, 4]
        assert later[0]["body"]["transcript"] != checkpoints[0]["body"]["transcript"]
    finally:
        grid.close()


def test_worker_with_an_unpinned_identity_is_rejected_at_the_handshake(
    mtls_identities: MutualTLSIdentities,
) -> None:
    # The session handshake already refuses a worker that cannot sign as the pinned
    # identity, so no activation is ever sent to it.
    grid = _Grid(mtls_identities, swap_identity=True)
    try:
        with pytest.raises(ProtocolError, match="pinned worker identity"):
            grid.call()
    finally:
        grid.close()


def test_worker_with_other_artifacts_is_rejected_at_the_handshake(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities, swap_artifacts=True)
    try:
        with pytest.raises(ProtocolError, match="artifacts other than the ones deployed"):
            grid.call()
    finally:
        grid.close()


def test_checkpoint_detects_traffic_the_client_did_not_send(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        grid.call()
        connection = next(iter(grid.runtime.connections.values()))
        connection.transcript.update(b"never sent", b"never received")
        with pytest.raises(ProtocolError, match="different number of calls"):
            grid.runtime.checkpoints()
    finally:
        grid.close()


def test_checkpoint_detects_altered_traffic_with_the_same_call_count(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        grid.call()
        connection = next(iter(grid.runtime.connections.values()))
        forged = SessionTranscript(connection.transcript.session_id)
        forged.update(b"x", b"y")
        connection.transcript = forged
        with pytest.raises(ProtocolError, match="different traffic"):
            grid.runtime.checkpoints()
    finally:
        grid.close()


def test_stored_checkpoint_fails_offline_verification_after_any_edit(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        grid.call()
        checkpoint = grid.runtime.checkpoints()[0]
    finally:
        grid.close()

    verify_checkpoint_signature(checkpoint)
    edited = {**checkpoint, "body": {**checkpoint["body"], "calls": 99}}
    with pytest.raises(ProtocolError, match="does not verify"):
        verify_checkpoint_signature(edited)
    other = HybridSigningKey.generate().public_key.hex()
    with pytest.raises(ProtocolError, match="does not verify"):
        verify_checkpoint_signature({**checkpoint, "identity_public": other})


def test_worker_without_an_identity_key_is_rejected_when_one_is_pinned(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities, signing=False)
    try:
        with pytest.raises(ProtocolError, match="pinned worker identity"):
            grid.call()
    finally:
        grid.close()


def test_checkpoint_name_cannot_be_used_for_an_operation(
    mtls_identities: MutualTLSIdentities,
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    private = HybridKemKey.generate()
    public = private.public_key
    with pytest.raises(ValueError, match="reserved"):
        FortressRoleWorker(
            {CHECKPOINT_OPERATION: encode_role_artifact(operation, 0, public)},
            private.to_bytes(),
            certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
            private_key=mtls_identities["server_key"],
            client_ca=mtls_identities["ca"],
            allow_unattested=True,
        )


def test_trace_verification_rejects_a_checkpoint_for_other_artifacts_than_the_manifest() -> None:
    from xurcroh.fortress.checkpoint import open_checkpoint, sign_checkpoint
    from xurcroh.receipts import TraceError, model_manifest_digest, sign_trace, verify_trace

    key = HybridSigningKey.generate()
    transcript = SessionTranscript("session")
    item = open_checkpoint(
        sign_checkpoint(key, transcript, artifacts="a" * 64, weight_index=0, nonce=bytes(32)),
        worker="w",
        identity_public=key.public_key,
        transcript=transcript,
        weight_index=0,
        nonce=bytes(32),
        expected_artifacts=None,
    )

    def trace(expected: str) -> dict:
        model: dict = {"role_artifacts": {"w": expected}}
        model["manifest_sha256"] = model_manifest_digest(model)
        return sign_trace(
            {"task_id": "t", "model": model, "transport": {"worker_checkpoints": [item]}},
            HybridSigningKey.generate(),
        )

    assert verify_trace(trace("a" * 64))["worker_checkpoints"] == 1
    with pytest.raises(TraceError, match="artifacts"):
        verify_trace(trace("b" * 64))


def test_checkpoint_reply_is_checked_against_the_pinned_identity_and_artifacts() -> None:
    from xurcroh.fortress.checkpoint import open_checkpoint, sign_checkpoint

    key = HybridSigningKey.generate()
    transcript = SessionTranscript("session")
    reply = sign_checkpoint(key, transcript, artifacts="a" * 64, weight_index=0, nonce=bytes(32))

    def check(**changes: object) -> dict:
        options: dict = {
            "worker": "w",
            "identity_public": key.public_key,
            "transcript": transcript,
            "weight_index": 0,
            "nonce": bytes(32),
            "expected_artifacts": "a" * 64,
        }
        return open_checkpoint(reply, **{**options, **changes})

    assert check()["artifacts_pinned"] is True
    with pytest.raises(ProtocolError, match="pinned worker identity"):
        check(identity_public=HybridSigningKey.generate().public_key)
    with pytest.raises(ProtocolError, match="artifacts other than the ones deployed"):
        check(expected_artifacts="b" * 64)
    with pytest.raises(ProtocolError, match="not fresh"):
        check(nonce=b"\x01" * 32)
    with pytest.raises(ProtocolError, match="another session"):
        check(transcript=SessionTranscript("other"))


def test_expired_session_is_replaced_by_a_fresh_handshake(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities)
    try:
        grid.call()
        first = {key: c.session.session_id for key, c in grid.runtime.connections.items()}
        for worker in grid.workers:
            for session in worker.sessions._sessions.values():
                session.expires_at = 0
            assert worker.sessions.active_count == 0  # every session has now expired

        grid.call()  # the worker refuses the old session; the client starts a new one

        second = {key: c.session.session_id for key, c in grid.runtime.connections.items()}
        assert all(first[key] != second[key] for key in first)
        checkpoints = grid.runtime.checkpoints()
        assert [item["body"]["calls"] for item in checkpoints] == [1, 1, 1, 1]
        assert all(not worker._transcripts.keys() & set(first.values()) for worker in grid.workers)
    finally:
        grid.close()
