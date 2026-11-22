from __future__ import annotations

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
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
                private = X25519PrivateKey.generate()
                public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                artifacts = {"op": encode_role_artifact(self.operation, weight_index, public)}
                key = HybridSigningKey.generate()
                worker = FortressRoleWorker(
                    artifacts,
                    private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
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


def test_checkpoint_from_an_unpinned_identity_is_rejected(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities, swap_identity=True)
    try:
        grid.call()
        with pytest.raises(ProtocolError, match="pinned worker identity"):
            grid.runtime.checkpoints()
    finally:
        grid.close()


def test_checkpoint_for_other_artifacts_is_rejected(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities, swap_artifacts=True)
    try:
        grid.call()
        with pytest.raises(ProtocolError, match="artifacts other than the ones deployed"):
            grid.runtime.checkpoints()
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


def test_roles_without_a_pinned_identity_report_no_checkpoint(
    mtls_identities: MutualTLSIdentities,
) -> None:
    grid = _Grid(mtls_identities, signing=False)
    try:
        grid.call()
        with pytest.raises(ProtocolError, match="checkpoint"):
            grid.runtime.checkpoints()
    finally:
        grid.close()


def test_checkpoint_name_cannot_be_used_for_an_operation(
    mtls_identities: MutualTLSIdentities,
) -> None:
    operation = compile_operation(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
    )
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    with pytest.raises(ValueError, match="reserved"):
        FortressRoleWorker(
            {CHECKPOINT_OPERATION: encode_role_artifact(operation, 0, public)},
            private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
            private_key=mtls_identities["server_key"],
            client_ca=mtls_identities["ca"],
            allow_unattested=True,
        )
