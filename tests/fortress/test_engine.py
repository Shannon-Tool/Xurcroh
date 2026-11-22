from __future__ import annotations

import json

import pytest

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import ConfigurationError
from xurcroh.fortress.engine import FortressEngine


def _engine(role_document: object) -> FortressEngine:
    definition = ModelDefinition(path="model", engine="fortress", modality="text")
    workers = [
        WorkerConfig(
            name="role-0",
            endpoint="quic://127.0.0.1:9444",
            labels={
                "operations": "*",
                "operation_roles": json.dumps({"op": role_document}),
                "recipient_public": "ab" * 1216,
            },
        )
    ]
    return FortressEngine("model", definition, workers)


def test_engine_rejects_non_integer_role_coordinates() -> None:
    engine = _engine([0.5, 1])
    with pytest.raises(ConfigurationError, match="role"):
        engine._roles("op")
    engine = _engine(["0", 1])
    with pytest.raises(ConfigurationError, match="role"):
        engine._roles("op")


def test_engine_accepts_integer_role_coordinates() -> None:
    engine = _engine([0, 1])
    roles = engine._roles("op")
    assert len(roles) == 1
    assert roles[0].activation_index == 0
    assert roles[0].weight_index == 1
