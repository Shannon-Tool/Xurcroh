from __future__ import annotations

import struct
from pathlib import Path

import pytest
from tests.formats.gguf_utils import write_gguf

from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats import open_checkpoint


def test_gguf_rejects_excessive_counts_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "counts.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 2**63, 0))
    with pytest.raises(ModelLoadError, match="tensor count"):
        open_checkpoint(path)


def test_gguf_rejects_out_of_bounds_tensor(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "bounds.gguf", [("weight", (1, 4), 0, bytes(16))])
    payload = bytearray(path.read_bytes())
    payload[-1] = 0
    path.write_bytes(payload[:-20])
    with pytest.raises(ModelLoadError, match="outside"):
        open_checkpoint(path)


def test_gguf_rejects_unknown_type_code(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "unknown.gguf", [("weight", (1, 4), 999, bytes(4))])
    with pytest.raises(CapabilityError, match="unsupported GGML type"):
        open_checkpoint(path)


def test_gguf_rejects_invalid_alignment(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "alignment.gguf",
        [("weight", (1, 4), 0, bytes(16))],
        alignment=7,
    )
    with pytest.raises(ModelLoadError, match="alignment"):
        open_checkpoint(path)


def test_gguf_rejects_duplicate_tensor_names(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "duplicates.gguf",
        [("weight", (1, 4), 0, bytes(16)), ("weight", (1, 4), 0, bytes(16))],
    )
    with pytest.raises(ModelLoadError, match="duplicate tensor"):
        open_checkpoint(path)


def test_gguf_rejects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "short.gguf"
    path.write_bytes(b"GGUF\x03")
    with pytest.raises(ModelLoadError, match="truncated"):
        open_checkpoint(path)
