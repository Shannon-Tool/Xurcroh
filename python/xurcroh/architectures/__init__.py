from __future__ import annotations

from pathlib import Path

from xurcroh.architectures.base import ArchitectureAdapter
from xurcroh.architectures.llama import open_gguf_architecture
from xurcroh.architectures.quantized import QuantizedArchitectureAdapter
from xurcroh.formats import open_checkpoint

__all__ = ["ArchitectureAdapter", "open_architecture"]


def open_architecture(path: str | Path, source_format: str = "auto") -> ArchitectureAdapter:
    reader = open_checkpoint(path, source_format)
    try:
        if reader.source_format == "gguf":
            return open_gguf_architecture(reader)
        if reader.source_format in {"exl2", "exl3"}:
            return QuantizedArchitectureAdapter(reader)
        return QuantizedArchitectureAdapter(reader)
    except Exception:
        reader.close()
        raise
