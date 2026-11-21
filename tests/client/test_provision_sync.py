from __future__ import annotations

import threading

import pytest

from xurcroh.config import ClusterConfig, ModelDefinition
from xurcroh.provisioning import provision_cluster_sync


def _cluster_and_models() -> tuple[ClusterConfig, tuple[tuple[str, ModelDefinition], ...]]:
    cluster = ClusterConfig.normalize({})
    models = (("tiny", ModelDefinition(path="/nonexistent/model")),)
    return cluster, models


def test_provision_cluster_sync_spawns_no_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster, models = _cluster_and_models()

    def forbidden_thread(*args: object, **kwargs: object) -> threading.Thread:
        raise AssertionError("provision_cluster_sync must not spawn a thread")

    class FakeDeployment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> FakeDeployment:
            return self

    monkeypatch.setattr(threading, "Thread", forbidden_thread)
    monkeypatch.setattr(
        "xurcroh.fortress.deployment.FortressDeployment", FakeDeployment
    )
    provision_cluster_sync(cluster, models, controller="quic://127.0.0.1:0")


def test_provision_cluster_sync_returns_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster, models = _cluster_and_models()

    class FakeDeployment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> FakeDeployment:
            return self

    monkeypatch.setattr(
        "xurcroh.fortress.deployment.FortressDeployment", FakeDeployment
    )
    deployment = provision_cluster_sync(cluster, models, controller="quic://127.0.0.1:0")
    assert isinstance(deployment, FakeDeployment)


def test_provision_cluster_sync_propagates_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster, models = _cluster_and_models()
    failure = RuntimeError("deployment failed")

    class FailingDeployment:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> object:
            raise failure

    monkeypatch.setattr(
        "xurcroh.fortress.deployment.FortressDeployment", FailingDeployment
    )
    with pytest.raises(RuntimeError) as excinfo:
        provision_cluster_sync(cluster, models, controller="quic://127.0.0.1:0")
    assert excinfo.value is failure
