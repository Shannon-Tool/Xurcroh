from __future__ import annotations

from pathlib import Path

from shards.identities import StageIdentities
from xurcroh.cli import main


def test_compile_cli_creates_encrypted_artifacts(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "compiled"
    identities = StageIdentities.create(2)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    arguments = [
        "xurcroh",
        "compile",
        "--checkpoint",
        str(tiny_llama_path),
        "--output",
        str(output),
        "--stages",
        "2",
        "--boundary-key-env",
        "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
    ]
    for recipient in identities.recipients:
        arguments.extend(
            ["--stage-recipient", f"{recipient.stage_index}={recipient.public_key.hex()}"]
        )
    monkeypatch.setattr("sys.argv", arguments)
    main()
    assert (output / "manifest.json").is_file()
    assert (output / "client-boundary.xurcroh").is_file()
    assert len(list(output.glob("stage-*.xurcroh"))) == 2
