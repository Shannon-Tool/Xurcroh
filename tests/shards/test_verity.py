from __future__ import annotations

from pathlib import Path

from xurcroh.shards.verity import VerityPackager


def test_verity_packager_invokes_real_tool_commands(tmp_path: Path) -> None:
    mkfs = tmp_path / "mkfs.erofs"
    verity = tmp_path / "veritysetup"
    shard = tmp_path / "stage.xurcroh"
    shard.write_bytes(b"encrypted-test-shard")
    mkfs.write_text('#!/bin/sh\ncp "$2/stage.xurcroh" "$1.image-contents"\n')
    verity.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = format ]; then echo 'Root hash: 001122aabbcc'; fi\n"
        'if [ "$1" = verify ]; then test "$4" = 001122aabbcc; fi\n'
    )
    mkfs.chmod(0o755)
    verity.chmod(0o755)
    packager = VerityPackager(mkfs_erofs=mkfs, veritysetup=verity)
    root_hash = packager.package(
        shard,
        tmp_path / "stage.erofs",
        tmp_path / "stage.verity",
    )
    assert root_hash == "001122aabbcc"
    packager.verify(tmp_path / "stage.erofs", tmp_path / "stage.verity", root_hash)
    assert (tmp_path / "stage.erofs.image-contents").read_bytes() == b"encrypted-test-shard"
