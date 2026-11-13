from __future__ import annotations

import pytest
from scripts.fortress.remote_ladder import remote_command, worker_configuration


def test_remote_worker_configuration_rewrites_key_without_mutating_input() -> None:
    source = {"workers": [{"name": "a", "ssh_key": "local"}]}
    result = worker_configuration(source, "/remote/key")
    assert result["workers"][0]["ssh_key"] == "/remote/key"
    assert source["workers"][0]["ssh_key"] == "local"
    with pytest.raises(ValueError, match="workers"):
        worker_configuration({"workers": []}, "/remote/key")


def test_remote_ladder_command_preserves_revision_and_persistent_log() -> None:
    command = remote_command("/srv/gateway", "a" * 40)
    assert "XURCROH_SOURCE_REVISION=" + "a" * 40 in command
    assert "set -o pipefail" in command
    assert "tee /srv/gateway/run.log" in command
