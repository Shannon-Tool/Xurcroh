from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

VALUE_TYPES = {
    int: 10,
    float: 6,
    bool: 7,
    str: 8,
    list: 9,
}


def _string(value: str, endian: str) -> bytes:
    encoded = value.encode()
    return struct.pack(f"{endian}Q", len(encoded)) + encoded


def _value(value: Any, endian: str) -> bytes:
    if isinstance(value, bool):
        return struct.pack(f"{endian}I", 7) + bytes([value])
    if isinstance(value, int):
        return struct.pack(f"{endian}IQ", 10, value)
    if isinstance(value, float):
        return struct.pack(f"{endian}Id", 12, value)
    if isinstance(value, str):
        return struct.pack(f"{endian}I", 8) + _string(value, endian)
    if isinstance(value, list):
        if not value:
            return struct.pack(f"{endian}IIQ", 9, 8, 0)
        if all(isinstance(item, str) for item in value):
            return struct.pack(f"{endian}IIQ", 9, 8, len(value)) + b"".join(
                _string(item, endian) for item in value
            )
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return struct.pack(f"{endian}IIQ", 9, 10, len(value)) + b"".join(
                struct.pack(f"{endian}Q", item) for item in value
            )
    raise TypeError(type(value))


def write_gguf(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    *,
    metadata: dict[str, Any] | None = None,
    endian: str = "<",
    version: int = 3,
    alignment: int = 32,
) -> Path:
    values = {
        "general.architecture": "llama",
        "general.alignment": alignment,
        "general.quantization_version": 2,
        "llama.embedding_length": 4,
        "llama.block_count": 1,
        "llama.feed_forward_length": 8,
        "llama.context_length": 16,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 1,
        "llama.attention.layer_norm_rms_epsilon": 1e-6,
        "llama.rope.freq_base": 10000.0,
        **(metadata or {}),
    }
    count_format = "I" if version == 1 else "Q"
    header = b"GGUF" + struct.pack(f"{endian}I", version)
    header += struct.pack(f"{endian}{count_format * 2}", len(tensors), len(values))
    for key, value in values.items():
        header += _string(key, endian) + _value(value, endian)
    offset = 0
    infos = []
    payloads = []
    for name, shape, encoding, payload in tensors:
        info = _string(name, endian) + struct.pack(f"{endian}I", len(shape))
        info += b"".join(struct.pack(f"{endian}Q", size) for size in reversed(shape))
        info += struct.pack(f"{endian}IQ", encoding, offset)
        infos.append(info)
        padding = (-len(payload)) % alignment
        payloads.append(payload + bytes(padding))
        offset += len(payload) + padding
    body = b"".join(infos)
    padding = (-len(header) - len(body)) % alignment
    path.write_bytes(header + body + bytes(padding) + b"".join(payloads))
    return path
