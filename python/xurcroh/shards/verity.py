from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from xurcroh.errors import CompatibilityError, ShardIntegrityError


class VerityPackager:
    def __init__(
        self,
        *,
        mkfs_erofs: str | Path = "mkfs.erofs",
        veritysetup: str | Path = "veritysetup",
    ) -> None:
        self.mkfs_erofs = str(mkfs_erofs)
        self.veritysetup = str(veritysetup)

    def _run(self, arguments: list[str]) -> str:
        try:
            result = subprocess.run(  # noqa: S603 - fixed operator-selected tools, no shell
                arguments, capture_output=True, text=True, check=False
            )
        except OSError as error:
            raise CompatibilityError(
                f"required verity tool is unavailable: {arguments[0]}"
            ) from error
        if result.returncode != 0:
            raise ShardIntegrityError(
                f"verity command failed with status {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout

    def package(self, shard: str | Path, image: str | Path, hash_tree: str | Path) -> str:
        shard_path = Path(shard)
        image_path = Path(image)
        hash_path = Path(hash_tree)
        if not shard_path.is_file():
            raise ShardIntegrityError(f"missing shard for verity packaging: {shard_path}")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="xurcroh-erofs-") as staging:
            shutil.copyfile(shard_path, Path(staging) / shard_path.name)
            self._run([self.mkfs_erofs, str(image_path), staging])
        output = self._run([self.veritysetup, "format", str(image_path), str(hash_path)])
        match = re.search(r"Root hash:\s*([0-9a-fA-F]+)", output)
        if match is None:
            raise ShardIntegrityError("veritysetup did not return a root hash")
        return match.group(1).lower()

    def verify(self, image: str | Path, hash_tree: str | Path, expected_root_hash: str) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]+", expected_root_hash):
            raise ShardIntegrityError("invalid dm-verity root hash")
        self._run([self.veritysetup, "verify", str(image), str(hash_tree), expected_root_hash])
