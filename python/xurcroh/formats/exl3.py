from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from xurcroh.errors import ModelLoadError
from xurcroh.formats.safetensors import SafetensorsReader


def tensor_core_permutation() -> np.ndarray:
    permutation = np.empty(256, dtype=np.int64)
    for tile in range(32):
        r0 = (tile % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = tile // 4
        c1 = c0 + 8
        permutation[tile * 8 : tile * 8 + 8] = [
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


def _window(words: np.ndarray, start: int) -> int:
    ring_size = len(words) * 32
    start %= ring_size
    index, offset = divmod(start, 32)
    merged = (int(words[index]) << 32) | int(words[(index + 1) % len(words)])
    return (merged >> (48 - offset)) & 0xFFFF


def _decode_codebook(value: int, codebook: str) -> np.float16:
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


def decode_trellis_tile(packed: np.ndarray, bitrate: float, codebook: str) -> np.ndarray:
    integer = int(bitrate)
    fractional = bitrate != integer
    words = packed.view(np.uint32)
    block_bits = 16 * integer + (8 if fractional else 0)
    values = []
    for index in range(256):
        if fractional:
            end = (index // 16) * block_bits
            for position in range(index % 16 + 1):
                end += integer + ((0xAAAA >> position) & 1)
        else:
            end = (index + 1) * integer
        values.append(_decode_codebook(_window(words, end - 16), codebook))
    permutation = np.argsort(tensor_core_permutation())
    return np.asarray(values, dtype=np.float16)[permutation].reshape(16, 16)


def _hadamard() -> torch.Tensor:
    matrix = torch.tensor([[1.0]], dtype=torch.float32)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(128)


def _signs(tensors: dict[str, torch.Tensor], prefix: str, side: str, expected: int) -> torch.Tensor:
    unpacked = tensors.get(f"{prefix}.{side}h")
    if unpacked is not None:
        values = unpacked.to(torch.float16).flatten()
    else:
        packed = tensors.get(f"{prefix}.{side}")
        if packed is None:
            raise ModelLoadError(f"EXL3 tensor is missing {side}/{side}h sign vector")
        bitfield = packed.to(torch.int16).view(torch.uint16).to(torch.int32)
        masks = 1 << torch.arange(16, dtype=torch.int32)
        values = (
            (1 - 2 * ((bitfield.unsqueeze(-1) & masks) > 0).to(torch.int16))
            .to(torch.float16)
            .flatten()
        )
    if values.numel() != expected:
        raise ModelLoadError("EXL3 sign-vector dimensions are inconsistent")
    return values


def decode_exl3(
    tensors: dict[str, torch.Tensor], prefix: str, columns: slice | None = None
) -> torch.Tensor:
    trellis = tensors.get(f"{prefix}.trellis")
    if trellis is None or trellis.ndim != 3 or trellis.dtype != torch.int16:
        raise ModelLoadError("EXL3 trellis must be a three-dimensional int16 tensor")
    if not trellis.shape[0] or not trellis.shape[1] or trellis.shape[1] % 8:
        raise ModelLoadError("EXL3 output tiles must contain complete 128-column blocks")
    words = trellis.shape[2]
    if words == 0 or words % 8:
        raise ModelLoadError("EXL3 trellis must contain complete 32-bit words")
    bitrate = words / 16
    integer = int(bitrate)
    fractional = bitrate != integer
    if integer < 1 or integer > 8 or (fractional and bitrate != integer + 0.5):
        raise ModelLoadError(f"unsupported EXL3 bitrate: {bitrate}")
    mcg = f"{prefix}.mcg" in tensors
    mul1 = f"{prefix}.mul1" in tensors
    if mcg and mul1:
        raise ModelLoadError("EXL3 codebook markers are mutually exclusive")
    if fractional and not mul1:
        raise ModelLoadError("fractional EXL3 bitrates require the mul1 codebook")
    if fractional and integer > 7:
        raise ModelLoadError("fractional EXL3 bitrate is outside the supported range")
    codebook = "mul1" if mul1 else "mcg" if mcg else "3inst"
    input_size = trellis.shape[0] * 16
    output_size = trellis.shape[1] * 16
    if input_size % 128:
        raise ModelLoadError("EXL3 input tiles must contain complete 128-row blocks")
    if columns is not None and columns.step not in {None, 1}:
        raise ValueError("EXL3 output-column slices must be contiguous")
    start, stop, _ = (0, output_size, 1) if columns is None else columns.indices(output_size)
    su = _signs(tensors, prefix, "su", input_size)
    sv = _signs(tensors, prefix, "sv", output_size)
    packed = trellis.detach().cpu().numpy()
    transform = _hadamard()
    restored_blocks = []
    first_block, last_block = start // 128, (stop - 1) // 128 if stop > start else -1
    if stop <= start:
        return torch.empty((0, input_size), dtype=torch.float16)
    for output_block in range(first_block, last_block + 1):
        block_start = output_block * 128
        restored = torch.empty((input_size, 128), dtype=torch.float16)
        for input_block in range(input_size // 128):
            rotated = torch.empty((128, 128), dtype=torch.float16)
            for tile_row in range(8):
                for tile_column in range(8):
                    values = decode_trellis_tile(
                        packed[
                            input_block * 8 + tile_row,
                            output_block * 8 + tile_column,
                        ],
                        bitrate,
                        codebook,
                    )
                    rotated[
                        tile_row * 16 : (tile_row + 1) * 16,
                        tile_column * 16 : (tile_column + 1) * 16,
                    ] = torch.from_numpy(values)
            left = (transform @ rotated.to(torch.float32)).to(torch.float16)
            left *= su[input_block * 128 : (input_block + 1) * 128, None]
            right = (left.to(torch.float32) @ transform).to(torch.float16)
            right *= sv[None, block_start : block_start + 128]
            restored[input_block * 128 : (input_block + 1) * 128] = right
        selected_start = max(start, block_start) - block_start
        selected_stop = min(stop, block_start + 128) - block_start
        restored_blocks.append(restored[:, selected_start:selected_stop])
    return torch.cat(restored_blocks, dim=1).T.contiguous()


class EXL3Reader(SafetensorsReader):
    source_format = "exl3"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.encodings = tuple(sorted({*self.encodings, "EXL3"}))

    def _decode_named(self, name: str, columns: slice | None = None) -> torch.Tensor:
        prefix = name.removesuffix(".weight").removesuffix(".trellis")
        if f"{prefix}.trellis" not in self.weight_map:
            tensor = SafetensorsReader.read_tensors(self, [name])[name]
            return tensor if columns is None else tensor[columns, :]
        names = [
            raw
            for raw in self.weight_map
            if raw.startswith(prefix + ".")
            and raw.rsplit(".", 1)[-1] in {"trellis", "su", "sv", "suh", "svh", "mcg", "mul1"}
        ]
        return decode_exl3(super().read_tensors(names), prefix, columns)

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        return {name: self._decode_named(name) for name in names}

    def read_operation(self, name: str, columns: slice | None = None) -> torch.Tensor:
        self._ensure_open()
        return self._decode_named(name, columns)
