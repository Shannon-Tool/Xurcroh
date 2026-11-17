from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

from xurcroh.errors import CapabilityError


class ReferenceDecoderBridge:
    def __init__(self, package: str) -> None:
        if package not in {"exllamav2", "exllamav3"}:
            raise ValueError(f"unsupported reference decoder package: {package}")
        self.package = package

    def identity(self) -> dict[str, str]:
        try:
            version = metadata.version(self.package)
            distribution = metadata.distribution(self.package)
        except metadata.PackageNotFoundError as error:
            raise CapabilityError(
                f"pinned reference decoder {self.package} is not installed"
            ) from error
        digest = hashlib.sha256()
        decoder_names = (
            ("linear.py", "ext.py") if self.package == "exllamav2" else ("exl3.py", "quantize.py")
        )
        files = sorted(
            path
            for path in distribution.files or []
            if str(path).startswith(self.package) and path.name in decoder_names
        )
        if not files:
            raise CapabilityError(f"cannot fingerprint {self.package} decoder files")
        for file in files:
            digest.update(str(file).encode() + b"\0")
            digest.update(Path(file.locate()).read_bytes())
        return {"package": self.package, "version": version, "revision": digest.hexdigest()}
