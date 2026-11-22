"""One digest naming the client build that is running.

The digest covers every Python source file of the ``xurcroh`` and ``Xurcroh`` packages
and the native ``xurcroh_quic`` module. The same digest is computed from the release
wheels, so a release can publish it and anyone can compare an installation against it:

    python -m xurcroh.build                 # this installation
    python -m xurcroh.build dist/*.whl      # the wheels of a release

A client reports this digest in every task trace. That report is the client's own
statement: software that has been modified can also be modified to report another
digest. Recomputing the digest from outside the running client is what shows which
build is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import zipfile
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath

_DOMAIN = b"xurcroh-client-build-v1"
_PACKAGES = ("xurcroh", "Xurcroh", "xurcroh_quic")
_NATIVE = "xurcroh_quic"
_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def _entry_name(path: str) -> str | None:
    """The name a file is hashed under, or ``None`` if it is not part of the build."""

    parts = PurePosixPath(path).parts
    if not parts or "__pycache__" in parts:
        return None
    if parts[-1].startswith(_NATIVE) and parts[-1].endswith(_NATIVE_SUFFIXES):
        # Installers place the native module at different depths; its file name is stable.
        return f"native/{parts[-1]}"
    if parts[0] in _PACKAGES and path.endswith(".py"):
        return path
    return None


def _digest(entries: dict[str, str]) -> str:
    listing = json.dumps(sorted(entries.items()), separators=(",", ":"))
    return hashlib.sha256(_DOMAIN + listing.encode()).hexdigest()


def installed_entries() -> dict[str, str]:
    """Hashed name and SHA-256 of every build file of the running installation."""

    entries: dict[str, str] = {}
    for package in _PACKAGES:
        spec = importlib.util.find_spec(package)
        if spec is None:
            continue
        files: list[tuple[str, Path]] = []
        for location in spec.submodule_search_locations or []:
            files.extend(
                (f"{package}/{file.relative_to(location).as_posix()}", file)
                for file in Path(location).rglob("*")
                if file.is_file()
            )
        if not spec.submodule_search_locations and spec.origin:
            files.append((Path(spec.origin).name, Path(spec.origin)))
        for path, file in files:
            name = _entry_name(path)
            if name is not None:
                entries[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return entries


@lru_cache(maxsize=1)
def installed_build_digest() -> str:
    return _digest(installed_entries())


def wheel_build_digest(wheels: Sequence[Path]) -> str:
    """The digest an installation of exactly these wheels reports."""

    entries: dict[str, str] = {}
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                name = None if info.is_dir() else _entry_name(info.filename)
                if name is None:
                    continue
                if name in entries:
                    raise ValueError(f"{name} is in more than one wheel")
                entries[name] = hashlib.sha256(archive.read(info)).hexdigest()
    if not entries:
        raise ValueError("the wheels contain no client build files")
    return _digest(entries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the digest of a Xurcroh client build")
    parser.add_argument(
        "wheels", nargs="*", type=Path, help="wheels of a release; default: this installation"
    )
    args = parser.parse_args(argv)
    print(wheel_build_digest(args.wheels) if args.wheels else installed_build_digest())


if __name__ == "__main__":
    main()
