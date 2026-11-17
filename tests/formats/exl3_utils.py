from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def tensor_core_perm() -> np.ndarray:
    permutation = np.empty(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        permutation[t * 8 : t * 8 + 8] = [
            r0 * 16 + c0,
            r0 * 16 + c1,
            r1 * 16 + c0,
            r1 * 16 + c1,
            r2 * 16 + c0,
            r2 * 16 + c1,
            r3 * 16 + c0,
            r3 * 16 + c1,
        ]
    return permutation


def pack_tile(indices: np.ndarray, bitrate: float) -> np.ndarray:
    integer = int(bitrate)
    fractional = bitrate != integer
    stream = 0
    total = 0
    for index, value in enumerate(indices):
        width = integer + (index % 2) if fractional else integer
        stream = (stream << width) | int(value)
        total += width
    words = []
    for offset in range(total // 32):
        shift = total - (offset + 1) * 32
        words.append((stream >> shift) & 0xFFFFFFFF)
    return np.array(words, dtype=np.uint32).view(np.int16)


def window(words: np.ndarray, start: int) -> int:
    count = len(words) * 32
    start %= count
    index, offset = divmod(start, 32)
    merged = (int(words[index]) << 32) | int(words[(index + 1) % len(words)])
    return (merged >> (48 - offset)) & 0xFFFF


def decode_codebook(value: int, codebook: str) -> np.float16:
    if codebook == "3inst":
        value = (value * 89226354 + 64248484) & 0xFFFFFFFF
    elif codebook == "mcg":
        value = (value * 0xCBAC1FED) & 0xFFFFFFFF
    if codebook in {"3inst", "mcg"}:
        value = (value & 0x8FFF8FFF) ^ 0x3B603B60
        low = np.array([value & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
        high = np.array([value >> 16], dtype=np.uint16).view(np.float16)[0]
        return np.float16(low + high)
    value = (value * 0x83DCD12D) & 0xFFFFFFFF
    byte_sum = sum((value >> shift) & 0xFF for shift in (0, 8, 16, 24))
    bits = np.array([(byte_sum + 0x6400) & 0xFFFF], dtype=np.uint16)
    value_half = bits.view(np.float16)[0]
    inverse = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0]
    bias = np.array([0xC931], dtype=np.uint16).view(np.float16)[0]
    return np.float16(np.float64(value_half) * np.float64(inverse) + np.float64(bias))


def decode_tile(packed: np.ndarray, bitrate: float, codebook: str) -> np.ndarray:
    integer = int(bitrate)
    fractional = bitrate != integer
    mask = 0xAAAA if fractional else 0
    words = packed.view(np.uint32)
    block_bits = 16 * integer + (8 if fractional else 0)
    values = []
    for index in range(256):
        if fractional:
            end = (index // 16) * block_bits
            for position in range(index % 16 + 1):
                end += integer + ((mask >> position) & 1)
        else:
            end = (index + 1) * integer
        values.append(decode_codebook(window(words, end - 16), codebook))
    return np.asarray(values, dtype=np.float16)[np.argsort(tensor_core_perm())].reshape(16, 16)


def hadamard(dimension: int) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < dimension:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(dimension)


def reference_exl3(
    tensors: dict[str, torch.Tensor], prefix: str, bitrate: float, codebook: str
) -> torch.Tensor:
    trellis = tensors[f"{prefix}.trellis"].numpy()
    rows = trellis.shape[0] * 16
    columns = trellis.shape[1] * 16
    rotated = np.empty((rows, columns), dtype=np.float16)
    for row in range(trellis.shape[0]):
        for column in range(trellis.shape[1]):
            rotated[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16] = decode_tile(
                trellis[row, column], bitrate, codebook
            )
    transform = torch.from_numpy(hadamard(128).astype(np.float32))
    restored = np.empty_like(rotated)
    su = tensors[f"{prefix}.suh"]
    sv = tensors[f"{prefix}.svh"]
    for row in range(rows // 128):
        for column in range(columns // 128):
            block = torch.from_numpy(
                rotated[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128]
            )
            value = (transform @ block.to(torch.float32)).to(torch.float16)
            value *= su[row * 128 : (row + 1) * 128, None]
            value = (value.to(torch.float32) @ transform).to(torch.float16)
            value *= sv[None, column * 128 : (column + 1) * 128]
            restored[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128] = value.numpy()
    return torch.from_numpy(restored.T.copy())


def exl3_fixture(root: Path, bitrate: float, codebook: str) -> torch.Tensor:
    prefix = "model.layers.0.mlp.down_proj"
    integer = int(bitrate)
    generator = np.random.default_rng(23 + integer)
    tiles = []
    for _ in range(64):
        width = integer + 1 if bitrate != integer else integer
        indices = generator.integers(0, 2**width, 256, dtype=np.int64)
        if bitrate != integer:
            indices[0::2] &= (1 << integer) - 1
            indices[1::2] &= (1 << (integer + 1)) - 1
        else:
            indices &= (1 << integer) - 1
        tiles.append(pack_tile(indices, bitrate))
    trellis = np.stack(tiles).reshape(8, 8, -1)
    signs = np.ones(128, dtype=np.float16)
    signs[::11] = -1
    tensors = {
        f"{prefix}.trellis": torch.from_numpy(trellis.copy()),
        f"{prefix}.suh": torch.from_numpy(signs.copy()),
        f"{prefix}.svh": torch.from_numpy(signs[::-1].copy()),
    }
    if codebook in {"mcg", "mul1"}:
        tensors[f"{prefix}.{codebook}"] = torch.tensor(1, dtype=torch.int32)
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    save_file(tensors, str(root / "model.safetensors"))
    return reference_exl3(tensors, prefix, bitrate, codebook)
