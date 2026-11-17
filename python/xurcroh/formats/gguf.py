from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats.base import TensorDescriptor, TensorReader, fingerprint_source
from xurcroh.formats.gguf_metadata import gguf_to_transformers_config

MAX_RECORDS = 100_000
MAX_ARRAY_VALUES = 1_000_000
MAX_METADATA_DEPTH = 8
ByteOrder = Literal["<", ">"]


class GGUFEncoding(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42
    F8_E4M3 = 43


SCALARS = {
    GGUFEncoding.F32: (1, 4, "f4"),
    GGUFEncoding.F16: (1, 2, "f2"),
    GGUFEncoding.I8: (1, 1, "i1"),
    GGUFEncoding.I16: (1, 2, "i2"),
    GGUFEncoding.I32: (1, 4, "i4"),
    GGUFEncoding.I64: (1, 8, "i8"),
    GGUFEncoding.F64: (1, 8, "f8"),
    GGUFEncoding.F8_E4M3: (1, 1, "u1"),
}
CUSTOM_QUANT_SIZES = {
    GGUFEncoding.Q8_1: (32, 36),
    GGUFEncoding.Q1_0: (128, 18),
    GGUFEncoding.Q2_0: (64, 18),
}


@dataclass
class _Tensor:
    descriptor: TensorDescriptor
    ggml_shape: tuple[int, ...]
    data_offset: int
    n_bytes: int
    byte_order: ByteOrder


class _Parser:
    def __init__(self, data: np.memmap, byte_order: ByteOrder, version: int) -> None:
        self.data = data
        self.byte_order = byte_order
        self.version = version
        self.offset = 0
        self.values_read = 0

    def read(self, format_string: str) -> Any:
        size = struct.calcsize(format_string)
        end = self.offset + size
        if end > self.data.size:
            raise ModelLoadError("GGUF file is truncated")
        value = struct.unpack_from(format_string, self.data, self.offset)
        self.offset = end
        return value[0] if len(value) == 1 else value

    def integer(self, format_character: str) -> int:
        return int(self.read(self.byte_order + format_character))

    def string(self, maximum: int) -> str:
        length = self.integer("Q")
        if length > maximum:
            raise ModelLoadError(f"GGUF string length exceeds {maximum} bytes")
        end = self.offset + length
        if end > self.data.size:
            raise ModelLoadError("GGUF file is truncated")
        try:
            value = bytes(self.data[self.offset : end]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ModelLoadError("GGUF string is not valid UTF-8") from error
        self.offset = end
        return value

    def value(self, depth: int = 0, declared_type: int | None = None) -> Any:
        if depth > MAX_METADATA_DEPTH:
            raise ModelLoadError("GGUF metadata nesting is too deep")
        self.values_read += 1
        if self.values_read > MAX_ARRAY_VALUES:
            raise ModelLoadError("GGUF metadata contains too many values")
        value_type = self.integer("I") if declared_type is None else declared_type
        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            10: "Q",
            11: "q",
            12: "d",
        }
        if value_type == 7:
            return bool(self.integer("B"))
        if value_type == 8:
            return self.string(65_535)
        if value_type == 9:
            subtype = self.integer("I")
            length = self.integer("Q")
            if length > MAX_ARRAY_VALUES - self.values_read:
                raise ModelLoadError("GGUF metadata array is too large")
            return [self.value(depth + 1, subtype) for _ in range(length)]
        if value_type not in formats:
            raise ModelLoadError(f"unsupported GGUF metadata type: {value_type}")
        return self.read(self.byte_order + formats[value_type])


def _quant_size(encoding: GGUFEncoding) -> tuple[int, int]:
    if encoding in CUSTOM_QUANT_SIZES:
        return CUSTOM_QUANT_SIZES[encoding]
    if encoding in SCALARS:
        block, width, _ = SCALARS[encoding]
        return block, width
    import gguf

    return gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(encoding.value)]


def _parse_file(path: Path) -> tuple[dict[str, Any], list[_Tensor], np.memmap]:
    try:
        data = np.memmap(path, mode="r", dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ModelLoadError(f"cannot map GGUF checkpoint: {path}") from error
    try:
        if data.size < 12 or bytes(data[:4]) != b"GGUF":
            raise ModelLoadError("GGUF file is truncated or has invalid magic")
        little_version = struct.unpack_from("<I", data, 4)[0]
        big_version = struct.unpack_from(">I", data, 4)[0]
        byte_order: ByteOrder
        if little_version in {1, 2, 3}:
            byte_order, version = "<", little_version
        elif big_version == 3:
            byte_order, version = ">", big_version
        else:
            raise CapabilityError(f"unsupported GGUF version: {little_version}")
        parser = _Parser(data, byte_order, version)
        parser.offset = 8
        count_character = "I" if version == 1 else "Q"
        tensor_count = parser.integer(count_character)
        metadata_count = parser.integer(count_character)
        if tensor_count > MAX_RECORDS:
            raise ModelLoadError("GGUF tensor count exceeds the configured bound")
        if metadata_count > MAX_RECORDS:
            raise ModelLoadError("GGUF metadata count exceeds the configured bound")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = parser.string(65_535)
            if key in metadata:
                raise ModelLoadError(f"duplicate GGUF metadata key: {key}")
            metadata[key] = parser.value()
        alignment = int(metadata.get("general.alignment", 32))
        if alignment < 8 or alignment % 8 != 0 or alignment & (alignment - 1):
            raise ModelLoadError("GGUF alignment must be a power of two and a multiple of eight")
        descriptors: list[_Tensor] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            name = parser.string(64)
            if name in names:
                raise ModelLoadError(f"duplicate tensor name in GGUF shard: {name}")
            names.add(name)
            dimensions = parser.integer("I")
            if dimensions < 1 or dimensions > 4:
                raise ModelLoadError("GGUF tensor dimensions must contain 1-4 values")
            ggml_shape = tuple(parser.integer("Q") for _ in range(dimensions))
            if any(size == 0 for size in ggml_shape):
                raise ModelLoadError("GGUF tensor dimensions must be positive")
            raw_type = parser.integer("I")
            try:
                encoding = GGUFEncoding(raw_type)
            except ValueError as error:
                raise CapabilityError(f"unsupported GGML type code: {raw_type}") from error
            offset = parser.integer("Q")
            if offset % alignment:
                raise ModelLoadError(f"GGUF tensor offset is not aligned: {name}")
            block, width = _quant_size(encoding)
            elements = math.prod(ggml_shape)
            if ggml_shape[0] % block or elements * width % block:
                raise ModelLoadError(f"GGUF tensor shape is invalid for {encoding.name}: {name}")
            n_bytes = elements * width // block
            descriptor = TensorDescriptor(
                name=name,
                shape=tuple(reversed(ggml_shape)),
                encoding=encoding.name,
                source=path.name,
                offset=offset,
            )
            descriptors.append(_Tensor(descriptor, ggml_shape, offset, n_bytes, byte_order))
        data_start = (parser.offset + alignment - 1) // alignment * alignment
        if data_start > data.size:
            raise ModelLoadError("GGUF tensor data starts outside the file")
        for tensor in descriptors:
            tensor.data_offset += data_start
            if tensor.data_offset < data_start or tensor.data_offset + tensor.n_bytes > data.size:
                raise ModelLoadError(
                    f"GGUF tensor data lies outside the file: {tensor.descriptor.name}"
                )
        return metadata, descriptors, data
    except Exception:
        data._mmap.close()  # type: ignore[attr-defined]
        raise


class GGUFReader(TensorReader):
    source_format = "gguf"

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._maps: list[np.memmap] = []
        self._tensors: dict[str, _Tensor] = {}
        try:
            metadata, tensors, data = _parse_file(path)
            self._maps.append(data)
            files = [path]
            split_count = int(metadata["split.count"]) if "split.count" in metadata else 1
            if split_count > 1:
                split_number = int(metadata["split.no"]) if "split.no" in metadata else -1
                if split_number != 0:
                    raise ModelLoadError("open the first GGUF shard (split.no=0)")
                expected = int(metadata["split.tensors.count"])
                for index in range(1, split_count):
                    name = re.sub(
                        r"-\d{5}-of-\d{5}(?=\.gguf$)",
                        f"-{index + 1:05d}-of-{split_count:05d}",
                        path.name,
                    )
                    if name == path.name:
                        raise ModelLoadError(
                            "sharded GGUF path does not follow the shard naming convention"
                        )
                    shard = path.with_name(name)
                    shard_metadata, shard_tensors, shard_data = _parse_file(shard)
                    self._maps.append(shard_data)
                    files.append(shard)
                    if (
                        int(shard_metadata["split.no"]) != index
                        or int(shard_metadata["split.count"]) != split_count
                    ):
                        raise ModelLoadError("GGUF shard indexes are inconsistent")
                    tensors.extend(shard_tensors)
                if len(tensors) != expected:
                    raise ModelLoadError("GGUF shards do not contain the declared tensor count")
            self._files = files
            self._metadata = metadata
            for tensor in tensors:
                name = tensor.descriptor.name
                if name in self._tensors:
                    raise CapabilityError(f"duplicate tensor across GGUF shards: {name}")
                self._tensors[name] = tensor
            self.architecture = str(metadata.get("general.architecture", "unknown"))
            self.encodings = tuple(sorted({tensor.descriptor.encoding for tensor in tensors}))
            self._config = gguf_to_transformers_config(metadata)
        except Exception:
            for data in self._maps:
                data._mmap.close()  # type: ignore[attr-defined]
            self._maps.clear()
            raise

    @property
    def config(self) -> dict[str, Any]:
        self._ensure_open()
        return self._config

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return self._metadata

    @property
    def fingerprint(self) -> str:
        self._ensure_open()
        return fingerprint_source(self.path.parent, self._files)

    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return tuple(self._tensors)

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        return {name: self._tensor(name).descriptor.shape for name in names}

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        return {name: self._decode(self._tensor(name)) for name in names}

    def read_operation(self, name: str, columns: slice | None = None) -> torch.Tensor:
        self._ensure_open()
        tensor = self._tensor(name)
        if len(tensor.descriptor.shape) != 2:
            raise ModelLoadError(f"checkpoint operation is not a matrix: {name}")
        if columns is not None and columns.step not in {None, 1}:
            raise ValueError("GGUF output-column slices must be contiguous")
        return self._decode(tensor, columns)

    def _tensor(self, name: str) -> _Tensor:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise ModelLoadError(f"checkpoint tensor is missing: {name}") from error

    def _decode(self, tensor: _Tensor, columns: slice | None = None) -> torch.Tensor:
        descriptor = tensor.descriptor
        shape = descriptor.shape
        data = self._maps[self._files.index(self.path.parent / descriptor.source)]
        row_start, row_stop = 0, shape[0]
        if columns is not None:
            row_start, row_stop, _ = columns.indices(shape[0])
        row_width = tensor.n_bytes // shape[0]
        start = tensor.data_offset + row_start * row_width
        stop = tensor.data_offset + row_stop * row_width
        raw = np.array(data[start:stop], copy=True)
        selected_shape = (row_stop - row_start, *shape[1:])
        encoding = GGUFEncoding[descriptor.encoding]
        if encoding in SCALARS and encoding != GGUFEncoding.F8_E4M3:
            _, _, numpy_type = SCALARS[encoding]
            dtype = np.dtype(numpy_type).newbyteorder(tensor.byte_order)
            values = raw.view(dtype).reshape(selected_shape)
            native = values.astype(dtype.newbyteorder("="), copy=True)
            return torch.from_numpy(native)
        block, width = _quant_size(encoding)
        byte_shape = (*selected_shape[:-1], selected_shape[-1] // block * width)
        raw = raw.reshape(byte_shape)
        if tensor.byte_order == ">":
            raise CapabilityError("big-endian quantized GGUF tensors are not supported yet")
        values = self._dequantize(raw, encoding)
        return torch.from_numpy(np.asarray(values).reshape(selected_shape).copy())

    def _dequantize(self, raw: np.ndarray, encoding: GGUFEncoding) -> np.ndarray:
        if encoding in {GGUFEncoding.Q1_0, GGUFEncoding.Q2_0}:
            scale, packed = np.hsplit(raw, [2])
            scale = scale.view(np.float16).astype(np.float32)
            if encoding == GGUFEncoding.Q1_0:
                bits = np.unpackbits(packed, axis=-1, bitorder="little")
                return np.where(bits != 0, scale, -scale).astype(np.float32)
            values = packed.reshape(*packed.shape[:-1], -1, 1) >> np.array(
                [0, 2, 4, 6], dtype=np.uint8
            )
            values = (values & 3).reshape(*packed.shape[:-1], -1).astype(np.int8) - 1
            return scale * values.astype(np.float32)
        if encoding == GGUFEncoding.F8_E4M3:
            bits = raw.astype(np.uint8)
            sign = np.where(bits & 0x80, np.float32(-1), np.float32(1))
            magnitude = bits & 0x7F
            exponent = magnitude >> 3
            mantissa = magnitude & 7
            values = np.where(
                exponent == 0,
                np.ldexp(mantissa.astype(np.float32), -9),
                np.ldexp(1 + mantissa.astype(np.float32) / 8, exponent.astype(np.int32) - 7),
            )
            return np.where(magnitude == 0x7F, np.nan, sign * values).astype(np.float32)
        if encoding == GGUFEncoding.Q8_1:
            scale = raw[:, :2].view(np.float16).astype(np.float32)
            return scale * raw[:, 4:].view(np.int8).astype(np.float32)
        if encoding == GGUFEncoding.Q8_K:
            scale = raw[:, :4].view(np.float32)
            return scale * raw[:, 4:260].view(np.int8).astype(np.float32)
        import gguf

        try:
            return gguf.dequantize(raw, gguf.GGMLQuantizationType(encoding.value))
        except (NotImplementedError, ValueError) as error:
            raise CapabilityError(f"GGUF encoding is not decodable: {encoding.name}") from error

    def close(self) -> None:
        if not self._closed:
            for data in self._maps:
                data._mmap.close()  # type: ignore[attr-defined]
            self._maps.clear()
        super().close()
