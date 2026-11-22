"""Cover traffic must look like real traffic to a worker and to a watcher on the path."""

from __future__ import annotations

import itertools
import secrets
import threading
import time

import pytest
import torch
from tests.fortress.conftest import MutualTLSIdentities

from xurcroh.fortress.arithmetic import activation_row_seed, seeded_split_secret
from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.transport import (
    FortressGridRuntime,
    MutualTLSIdentity,
    RoleEndpoint,
    _RoleTransport,
    encode_role_artifact,
)
from xurcroh.fortress.worker import FortressRoleWorker
from xurcroh.kem import HybridKemKey


class _Grid:
    """Four in-process role workers behind one grid runtime, with seeded activations."""

    def __init__(self, mtls: MutualTLSIdentities, **runtime_options: object) -> None:
        self.operation = compile_operation(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), threshold=2, slice_count=3
        )
        identity = MutualTLSIdentity(
            certificates=(mtls["client_certificate"], mtls["ca"]),
            private_key=mtls["client_key"],
            pinned_ca=mtls["ca"],
        )
        self.workers: dict[tuple[int, int], FortressRoleWorker] = {}
        roles = []
        for activation_index in range(2):
            for weight_index in range(2):
                private = HybridKemKey.generate()
                worker = FortressRoleWorker(
                    {"op": encode_role_artifact(self.operation, weight_index, private.public_key)},
                    private.to_bytes(),
                    certificates=(mtls["server_certificate"], mtls["ca"]),
                    private_key=mtls["server_key"],
                    client_ca=mtls["ca"],
                    allow_unattested=True,
                )
                worker.start()
                self.workers[(activation_index, weight_index)] = worker
                roles.append(
                    RoleEndpoint(
                        activation_index,
                        weight_index,
                        worker.address,
                        private.public_key,
                        f"role-{activation_index}-{weight_index}",
                    )
                )
        self.runtime = FortressGridRuntime(roles, identity, **runtime_options)  # type: ignore[arg-type]
        self.client = self.runtime.client(self.operation, "op")

    def call(self) -> torch.Tensor:
        """One real call, built the way ``FortressLinear.forward`` builds it."""

        seed = secrets.token_bytes(32)
        shares = seeded_split_secret(
            torch.tensor([5, 7], dtype=torch.int64), seed=activation_row_seed(seed, 0), count=2
        ).shares
        return self.client.execute(shares, request_id="op", activation_seed=seed)

    def close(self) -> None:
        self.runtime.close()
        for worker in self.workers.values():
            worker.close()


def _wait_for(condition, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition was not reached in time"
        time.sleep(0.01)


def test_a_worker_cannot_tell_a_cover_call_from_a_real_call(
    mtls_identities: MutualTLSIdentities, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[int, list[bytes]] = {}
    wire: list[tuple[bytes, int]] = []
    decode = FortressRoleWorker._decode_activation
    exchange = _RoleTransport._exchange

    def spy_decode(self: FortressRoleWorker, payload: bytes):  # type: ignore[no-untyped-def]
        seen.setdefault(id(self), []).append(bytes(payload))
        return decode(self, payload)

    def spy_exchange(self: _RoleTransport, request_hash: bytes, sequence: int, inner: bytes):  # type: ignore[no-untyped-def]
        wire.append((request_hash, sequence))
        return exchange(self, request_hash, sequence, inner)

    monkeypatch.setattr(FortressRoleWorker, "_decode_activation", spy_decode)
    monkeypatch.setattr(_RoleTransport, "_exchange", spy_exchange)
    grid = _Grid(mtls_identities, cover_interval=0.02)
    try:
        for _ in range(4):
            grid.call()
        _wait_for(lambda: grid.client.cover_request_count >= 16)
        real, cover = grid.client.request_count, grid.client.cover_request_count
    finally:
        grid.close()

    assert real == 16 and cover >= 16
    # After decryption, every payload a role saw has one form and one length: nothing
    # in it separates the cover calls from the real ones.
    for worker in grid.workers.values():
        payloads = seen[id(worker)]
        assert len(payloads) >= 8
        assert len({len(payload) for payload in payloads}) == 1
        assert len({payload[:4] == b"XF2\x01" for payload in payloads}) == 1
    # Roles that hold a seeded activation share receive the seed form for cover too.
    assert all(payload[:4] == b"XF2\x01" for payload in seen[id(grid.workers[(0, 0)])])
    # The identifier and sequence number on each frame, which a relay also sees, carry
    # no pattern: every call has a fresh identifier and the same sequence number.
    identifiers = [identifier for identifier, _ in wire]
    assert len(set(identifiers)) == len(identifiers)
    assert {sequence for _, sequence in wire} == {0}


def test_a_failed_cover_call_does_not_stop_real_requests(
    mtls_identities: MutualTLSIdentities, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = _Grid(mtls_identities, cover_interval=0.02)
    try:
        _wait_for(lambda: grid.client.cover_request_count >= 4)
        execute = _RoleTransport.execute
        failures = {"left": 2}

        def flaky(self: _RoleTransport, *args: object, cover: bool = False, **kwargs: object):  # type: ignore[no-untyped-def]
            if cover and failures["left"] > 0:
                failures["left"] -= 1
                raise ValueError("connection lost")
            return execute(self, *args, cover=cover, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_RoleTransport, "execute", flaky)
        _wait_for(lambda: grid.runtime.cover_failure_count >= 1)
        assert torch.equal(grid.call(), grid.call())
        sent = grid.client.cover_request_count
        _wait_for(lambda: grid.client.cover_request_count > sent + 4)
    finally:
        grid.close()


def test_constant_rate_sends_one_call_per_slot_with_or_without_a_request(
    mtls_identities: MutualTLSIdentities, monkeypatch: pytest.MonkeyPatch
) -> None:
    interval = 0.1
    sends: list[float] = []
    lock = threading.Lock()
    grid = _Grid(mtls_identities, cover_interval=interval, constant_rate=True)
    original = _RoleTransport._exchange

    def timed(self: _RoleTransport, request_hash: bytes, sequence: int, inner: bytes):  # type: ignore[no-untyped-def]
        if self.role.provider_id == "role-0-0":
            with lock:
                sends.append(time.monotonic())
        return original(self, request_hash, sequence, inner)

    monkeypatch.setattr(_RoleTransport, "_exchange", timed)
    try:
        # The first call on each link also carries the one-off session handshake frame.
        _wait_for(lambda: grid.client.cover_request_count >= 4)
        with lock:
            sends.clear()
        time.sleep(1.5)
        with lock:
            idle = list(sends)
            sends.clear()
        started = time.monotonic()
        real = 0
        while time.monotonic() - started < 1.5:
            grid.call()
            real += 1
        with lock:
            busy = list(sends)
    finally:
        grid.close()

    def gaps(moments: list[float]) -> list[float]:
        return [later - earlier for earlier, later in itertools.pairwise(moments)]

    assert real >= 5
    # The watcher sees the same schedule in both phases: never two calls closer than a
    # slot, and the same number of calls per second whether or not a request is running.
    for moments in (idle, busy):
        assert len(moments) >= 8
        assert min(gaps(moments)) >= interval * 0.7
    idle_rate = (len(idle) - 1) / (idle[-1] - idle[0])
    busy_rate = (len(busy) - 1) / (busy[-1] - busy[0])
    assert abs(idle_rate - busy_rate) / idle_rate < 0.25
    # Real calls used slots rather than adding to them.
    assert real <= len(busy) + 1
