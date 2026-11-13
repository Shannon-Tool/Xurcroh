from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from xurcroh.errors import ModelLoadError


def checkpoint_metadata(
    checkpoint: str | Path,
    *,
    repository: str,
    revision: str,
    allow_download: bool,
    license_name: str | None = None,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("checkpoint revision must be an immutable 40-character commit hash")
    root = Path(checkpoint)
    if not (root / "config.json").is_file():
        if not allow_download:
            raise ModelLoadError(f"trusted checkpoint is unavailable: {root}")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=root,
            local_files_only=False,
        )
    if not (root / "config.json").is_file():
        raise ModelLoadError(f"checkpoint has no config.json: {root}")
    license_text = None
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        path = root / name
        if path.is_file():
            license_text = path.read_text()
            break
    if license_name is None:
        if license_text is None:
            raise ModelLoadError(
                "checkpoint license file is missing; review and supply license_name"
            )
        if "Apache License" in license_text and "Version 2.0" in license_text:
            license_name = "Apache-2.0"
        else:
            license_name = license_text.strip().splitlines()[0]
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts
    )
    if not files:
        raise ModelLoadError("checkpoint directory is empty")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return {
        "repository": repository,
        "revision": revision,
        "license": license_name,
        "checkpoint_fingerprint": digest.hexdigest(),
        "checkpoint_path": str(root.resolve()),
    }
