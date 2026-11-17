#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import gguf
import numpy as np


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata(key: str, value: int | str) -> bytes:
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    return _string(key) + struct.pack("<IQ", 10, value)


def write_fixture(path: Path, name: str, logical: np.ndarray, kind: int, payload: bytes) -> None:
    metadata = {
        "general.architecture": "llama",
        "general.alignment": 32,
        "general.quantization_version": 2,
        "llama.embedding_length": logical.shape[-1],
        "llama.block_count": 1,
        "llama.feed_forward_length": logical.shape[-1],
        "llama.context_length": 16,
        "llama.attention.head_count": 1,
        "llama.attention.layer_norm_rms_epsilon": 1,
    }
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    header += b"".join(_metadata(key, value) for key, value in metadata.items())
    info = _string(name) + struct.pack("<I", logical.ndim)
    info += b"".join(struct.pack("<Q", size) for size in reversed(logical.shape))
    info += struct.pack("<IQ", kind, 0)
    padding = (-len(header) - len(info)) % 32
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + info + bytes(padding) + payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tiny public GGUF decoder fixtures")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    write_fixture(args.output / "f32.gguf", "weight", source, 0, source.tobytes())
    quantized = gguf.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    write_fixture(args.output / "q4_0.gguf", "weight", source, 2, quantized.tobytes())
    q1_source = np.linspace(-1, 1, 128, dtype=np.float32).reshape(1, 128)
    q1 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b01010101] * 16)
    write_fixture(args.output / "q1_0.gguf", "weight", q1_source, 41, q1)
    q2_source = np.linspace(-1, 1, 64, dtype=np.float32).reshape(1, 64)
    q2 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b11100100] * 16)
    write_fixture(args.output / "q2_0.gguf", "weight", q2_source, 42, q2)
    f8_source = np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float32)
    write_fixture(args.output / "f8.gguf", "weight", f8_source, 43, bytes([0, 0x38, 0xB8, 0x34]))


if __name__ == "__main__":
    main()
