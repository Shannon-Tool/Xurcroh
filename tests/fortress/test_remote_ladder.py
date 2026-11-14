from __future__ import annotations

from types import SimpleNamespace

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
    download = remote_command("/srv/gateway", "a" * 40, download_checkpoint=True)
    assert "scripts/fortress/prepare_checkpoint.py" in download
    assert "--allow-download" in download
    smoke = remote_command("/srv/gateway", "a" * 40, smoke=True)
    assert "scripts/fortress/grid_smoke.py" in smoke
    assert "scripts/fortress/run_ladder.py" not in smoke
    assert "--defer-provider-teardown" in smoke


def test_remote_smoke_teardown_covers_gateway_bootstrap_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.fortress.remote_ladder as module

    args = SimpleNamespace(smoke=True, state=tmp_path / "state.json", output=tmp_path)
    monkeypatch.setattr(
        module,
        "_run",
        lambda _args: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    called = []
    monkeypatch.setattr(
        module,
        "destroy_ladder_instances",
        lambda store, uncapped: called.append(uncapped) or {"all_destroyed": True},
    )
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        module.run(args)
    assert called == [True]
