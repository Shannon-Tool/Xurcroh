from __future__ import annotations

from pathlib import Path

import torch

from xurcroh.errors import ModelLoadError
from xurcroh.formats.exl_common import unpack_4bit_rows, unpack_columns, validate_permutation
from xurcroh.formats.safetensors import SafetensorsReader


def decode_exl2(tensors: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    required = ("q_weight", "q_scale", "q_scale_max", "q_groups")
    missing = [name for name in required if f"{prefix}.{name}" not in tensors]
    if missing:
        raise ModelLoadError(f"EXL2 tensor is missing fields: {', '.join(missing)}")
    q_weight = tensors[f"{prefix}.q_weight"]
    q_scale = tensors[f"{prefix}.q_scale"]
    q_scale_max = tensors[f"{prefix}.q_scale_max"]
    q_groups = tensors[f"{prefix}.q_groups"].flatten().to(torch.int64)
    if q_weight.ndim != 2 or q_groups.numel() % 2:
        raise ModelLoadError("EXL2 group metadata has inconsistent dimensions")
    groups = q_groups.reshape(-1, 2)
    if groups.shape[0] != q_scale_max.numel() or groups.shape[0] != q_scale.shape[0]:
        raise ModelLoadError("EXL2 group count does not match scale metadata")
    inverse = tensors.get(f"{prefix}.q_invperm")
    input_rows = inverse.numel() if inverse is not None else None
    decoded_groups = []
    packed_start = 0
    logical_start = 0
    for index, (bits_value, packed_row_value) in enumerate(groups):
        bits = int(bits_value)
        packed_row = int(packed_row_value)
        if bits not in {2, 3, 4, 5, 6, 8}:
            raise ModelLoadError(f"unsupported EXL2 bitrate: {bits}")
        if packed_row != packed_start:
            raise ModelLoadError("EXL2 group bitstream rows are not contiguous")
        if index + 1 < len(groups):
            next_packed = int(groups[index + 1, 1])
            if next_packed <= packed_row:
                raise ModelLoadError("EXL2 group rows must increase")
            packed_count = next_packed - packed_row
            rows = packed_count * 32 // bits
            if packed_count * 32 % bits:
                raise ModelLoadError("EXL2 group has a partial packed word")
        else:
            packed_count = q_weight.shape[0] - packed_start
            if packed_count <= 0:
                raise ModelLoadError("EXL2 final group is empty")
            if input_rows is None:
                if packed_count * 32 % bits:
                    raise ModelLoadError("EXL2 final group has an ambiguous row count")
                rows = packed_count * 32 // bits
            else:
                rows = input_rows - logical_start
        if rows <= 0 or (input_rows is not None and logical_start + rows > input_rows):
            raise ModelLoadError("EXL2 group row count is inconsistent")
        packed = q_weight[packed_start : packed_start + packed_count]
        expected_packed = (rows * bits + 31) // 32
        if packed.shape[0] != expected_packed:
            raise ModelLoadError("EXL2 group bitstream has an invalid size")
        unpacked = unpack_columns(packed, bits, rows)
        scale_values = unpack_4bit_rows(q_scale[index : index + 1], q_weight.shape[1])[0] + 1
        scale = scale_values.to(torch.float16).square() * q_scale_max[index].to(torch.float16)
        decoded = (unpacked.to(torch.float16) - 2 ** (bits - 1)) * scale
        decoded_groups.append(decoded)
        packed_start += packed_count
        logical_start += rows
    if packed_start != q_weight.shape[0] or (
        input_rows is not None and logical_start != input_rows
    ):
        raise ModelLoadError("EXL2 groups do not cover the complete matrix")
    result = torch.cat(decoded_groups).to(torch.float16)
    if inverse is not None:
        validate_permutation(inverse)
        permutation = torch.argsort(inverse.to(torch.int64))
        restored = torch.empty_like(result)
        restored[permutation] = result
        result = restored
    return result.T.contiguous()


def decode_gptq(
    tensors: dict[str, torch.Tensor], prefix: str, *, offset_qzeros: bool = False
) -> torch.Tensor:
    required = ("qweight", "qzeros", "scales")
    missing = [name for name in required if f"{prefix}.{name}" not in tensors]
    if missing:
        raise ModelLoadError(f"GPTQ tensor is missing fields: {', '.join(missing)}")
    qweight = tensors[f"{prefix}.qweight"]
    qzeros = tensors[f"{prefix}.qzeros"]
    scales = tensors[f"{prefix}.scales"]
    input_rows = qweight.shape[0] * 8
    output_columns = qweight.shape[1]
    if qzeros.shape != (scales.shape[0], output_columns // 8):
        raise ModelLoadError("GPTQ zero-point dimensions are inconsistent")
    quantized = unpack_columns(qweight, 4, input_rows)
    zeros = unpack_4bit_rows(qzeros, output_columns)
    if offset_qzeros:
        zeros -= 1
    group_index = tensors.get(f"{prefix}.g_idx")
    if group_index is None:
        if input_rows % scales.shape[0]:
            raise ModelLoadError("GPTQ group size is not integral")
        group_index = torch.arange(input_rows) // (input_rows // scales.shape[0])
    group_index = group_index.flatten().to(torch.int64)
    if group_index.numel() != input_rows or int(group_index.max()) >= scales.shape[0]:
        raise ModelLoadError("GPTQ group index is inconsistent")
    result = torch.empty((input_rows, output_columns), dtype=torch.float16)
    for group in range(scales.shape[0]):
        rows = group_index == group
        result[rows] = (quantized[rows] - zeros[group]).to(torch.float16) * scales[group]
    return result.T.contiguous()


class EXL2Reader(SafetensorsReader):
    source_format = "exl2"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        encodings = set(self.encodings)
        if any(name.endswith(".q_weight") for name in self.weight_map):
            encodings.add("EXL2")
        if any(name.endswith(".qweight") for name in self.weight_map):
            encodings.add("GPTQ")
        self.encodings = tuple(sorted(encodings))

    def _quantized_prefix(self, name: str) -> tuple[str, str] | None:
        prefix = name.removesuffix(".weight").removesuffix(".q_weight").removesuffix(".qweight")
        if f"{prefix}.q_weight" in self.weight_map:
            return prefix, "exl2"
        if f"{prefix}.qweight" in self.weight_map:
            return prefix, "gptq"
        return None

    def _decode_named(self, name: str, columns: slice | None = None) -> torch.Tensor:
        quantized = self._quantized_prefix(name)
        if quantized is None:
            tensor = SafetensorsReader.read_tensors(self, [name])[name]
            return tensor if columns is None else tensor[columns, :]
        prefix, kind = quantized
        names = [
            raw
            for raw in self.weight_map
            if raw.startswith(prefix + ".")
            and raw.rsplit(".", 1)[-1]
            in {
                "q_weight",
                "q_scale",
                "q_scale_max",
                "q_groups",
                "q_invperm",
                "qweight",
                "qzeros",
                "scales",
                "g_idx",
            }
        ]
        tensors = super().read_tensors(names)
        if kind == "exl2":
            result = decode_exl2(tensors, prefix)
        else:
            result = decode_gptq(
                tensors,
                prefix,
                offset_qzeros=bool(self.config.get("checkpoint_offset_qzeros", False)),
            )
        if columns is not None:
            if columns.step not in {None, 1}:
                raise ValueError("EXL2 output-column slices must be contiguous")
            result = result[columns, :]
        return result

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        return {name: self._decode_named(name) for name in names}

    def read_operation(self, name: str, columns: slice | None = None) -> torch.Tensor:
        self._ensure_open()
        return self._decode_named(name, columns)
