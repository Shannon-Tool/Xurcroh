from __future__ import annotations

import torch

from xurcroh.errors import ModelLoadError


def unpack_columns(packed: torch.Tensor, bits: int, rows: int) -> torch.Tensor:
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ModelLoadError(f"unsupported quantized bitrate: {bits}")
    if rows < 0:
        raise ModelLoadError("quantized row count cannot be negative")
    columns = packed.shape[1]
    unpacked = torch.empty((rows, columns), dtype=torch.int32)
    mask = (1 << bits) - 1
    for row in range(rows):
        bit = row * bits
        target = bit // 32
        shift = bit % 32
        if target >= packed.shape[0]:
            raise ModelLoadError("quantized group bitstream is truncated")
        value = packed[target].to(torch.int64) >> shift
        if shift + bits > 32:
            if target + 1 >= packed.shape[0]:
                raise ModelLoadError("quantized group crosses a truncated word boundary")
            value |= packed[target + 1].to(torch.int64) << (32 - shift)
        unpacked[row] = (value & mask).to(torch.int32)
    return unpacked


def unpack_4bit_rows(packed: torch.Tensor, columns: int) -> torch.Tensor:
    if columns % 8 or packed.shape[1] * 8 != columns:
        raise ModelLoadError("4-bit packed tensor has inconsistent dimensions")
    values = []
    for index in range(8):
        values.append((packed >> (index * 4)) & 0xF)
    return torch.stack(values, dim=-1).reshape(packed.shape[0], columns).to(torch.int32)


def validate_permutation(permutation: torch.Tensor) -> None:
    flattened = permutation.flatten().to(torch.int64)
    if flattened.numel() and not torch.equal(
        torch.sort(flattened).values, torch.arange(flattened.numel())
    ):
        raise ModelLoadError("quantized activation permutation is not bijective")
