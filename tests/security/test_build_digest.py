"""The client names its own build, and that name can be recomputed from the wheels."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

from xurcroh.build import installed_build_digest, installed_entries, wheel_build_digest


def _wheel(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def _installed_files() -> dict[str, bytes]:
    """The files of this installation, laid out as wheels lay them out."""

    files: dict[str, bytes] = {}
    for package in ("xurcroh", "Xurcroh", "xurcroh_quic"):
        spec = importlib.util.find_spec(package)
        assert spec is not None
        for location in spec.submodule_search_locations or []:
            for file in Path(location).rglob("*"):
                if file.is_file() and "__pycache__" not in file.parts:
                    relative = file.relative_to(location).as_posix()
                    files[f"{package}/{relative}"] = file.read_bytes()
        if not spec.submodule_search_locations and spec.origin:
            files[Path(spec.origin).name] = Path(spec.origin).read_bytes()
    return files


def test_installed_digest_equals_the_digest_of_the_wheels_it_came_from(tmp_path: Path) -> None:
    files = _installed_files()
    python = {name: data for name, data in files.items() if "quic" not in name}
    native = {name: data for name, data in files.items() if "quic" in name}
    assert python and native
    # Wheels also carry metadata and data files, which are not part of the build digest.
    python["xurcroh-0.1.0.dist-info/RECORD"] = b"anything"
    wheels = [_wheel(tmp_path / "client.whl", python), _wheel(tmp_path / "native.whl", native)]

    assert wheel_build_digest(wheels) == installed_build_digest()
    assert len(installed_build_digest()) == 64
    assert any(name.startswith("native/") for name in installed_entries())


def test_any_changed_or_added_source_file_changes_the_digest(tmp_path: Path) -> None:
    files = _installed_files()
    original = wheel_build_digest([_wheel(tmp_path / "a.whl", files)])

    changed = dict(files)
    changed["xurcroh/receipts.py"] += b"\n# backdoor\n"
    assert wheel_build_digest([_wheel(tmp_path / "b.whl", changed)]) != original

    added = dict(files)
    added["xurcroh/extra_module.py"] = b"print('hello')\n"
    assert wheel_build_digest([_wheel(tmp_path / "c.whl", added)]) != original


def test_wheels_naming_the_same_file_differently_are_refused(tmp_path: Path) -> None:
    first = _wheel(tmp_path / "a.whl", {"xurcroh/x.py": b"1"})
    second = _wheel(tmp_path / "b.whl", {"xurcroh/x.py": b"2"})
    with pytest.raises(ValueError, match="more than one wheel"):
        wheel_build_digest([first, second])
