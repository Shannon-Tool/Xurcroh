from __future__ import annotations

from pathlib import Path

from xurcroh.cli import main


def test_compile_cli_creates_encrypted_artifacts(
    tiny_llama_path: Path, tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "compiled"
    monkeypatch.setenv("TEST_XURCROH_MASTER_KEY", bytes(range(32)).hex())
    monkeypatch.setattr(
        "sys.argv",
        [
            "xurcroh",
            "compile",
            "--checkpoint",
            str(tiny_llama_path),
            "--output",
            str(output),
            "--stages",
            "2",
            "--master-key-env",
            "TEST_XURCROH_MASTER_KEY",
        ],
    )
    main()
    assert (output / "manifest.json").is_file()
    assert (output / "client-boundary.xurcroh").is_file()
    assert len(list(output.glob("stage-*.xurcroh"))) == 2
