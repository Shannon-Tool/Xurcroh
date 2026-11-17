from __future__ import annotations

import math
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from tests.formats.gguf_utils import write_gguf

from xurcroh.errors import CapabilityError
from xurcroh.formats import open_checkpoint
from xurcroh.formats.gguf_metadata import gguf_to_transformers_config
from xurcroh.formats.gguf_tokenizer import materialize_client_tokenizer

SUPPORTED_BY_UPSTREAM = [
    kind
    for kind in gguf.GGMLQuantizationType
    if kind.name not in {"Q8_1", "Q8_K", "I8", "I16", "I32", "I64", "F64", "Q1_0"}
]


@pytest.mark.parametrize("kind", SUPPORTED_BY_UPSTREAM, ids=lambda kind: kind.name)
def test_gguf_matches_upstream_dequantization(
    kind: gguf.GGMLQuantizationType, tmp_path: Path
) -> None:
    block, width = gguf.GGML_QUANT_SIZES[kind]
    logical_shape = (2, block)
    source = np.linspace(-1, 1, num=np.prod(logical_shape), dtype=np.float32).reshape(logical_shape)
    try:
        payload = gguf.quantize(source, kind).tobytes()
    except NotImplementedError:
        payload = (
            np.random.default_rng(kind.value)
            .integers(0, 256, size=np.prod(logical_shape) // block * width, dtype=np.uint8)
            .tobytes()
        )
    path = write_gguf(tmp_path / "model.gguf", [("weight", logical_shape, kind.value, payload)])
    with open_checkpoint(path) as reader:
        actual = reader.read_operation("weight").numpy()
        assert reader.encodings == (kind.name,)
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(2, width)
    expected = gguf.dequantize(raw, kind)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kind", "dtype"),
    [
        (gguf.GGMLQuantizationType.I8, np.int8),
        (gguf.GGMLQuantizationType.I16, np.int16),
        (gguf.GGMLQuantizationType.I32, np.int32),
        (gguf.GGMLQuantizationType.I64, np.int64),
        (gguf.GGMLQuantizationType.F64, np.float64),
    ],
)
def test_gguf_scalar_types(kind: gguf.GGMLQuantizationType, dtype: type, tmp_path: Path) -> None:
    values = np.arange(8, dtype=dtype).reshape(2, 4)
    path = write_gguf(
        tmp_path / "scalar.gguf", [("weight", values.shape, kind.value, values.tobytes())]
    )
    with open_checkpoint(path) as reader:
        actual = reader.read_operation("weight").numpy()
    np.testing.assert_array_equal(actual, values)


def test_gguf_q1_q2_and_f8_decoders(tmp_path: Path) -> None:
    q1 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b01010101] * 16)
    q2 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b11100100] * 16)
    f8 = bytes([0x00, 0x38, 0xB8, 0x7F])
    path = write_gguf(
        tmp_path / "new.gguf",
        [("q1", (1, 128), 41, q1), ("q2", (1, 64), 42, q2), ("f8", (1, 4), 43, f8)],
    )
    with open_checkpoint(path) as reader:
        q1_values = reader.read_operation("q1").numpy()
        q2_values = reader.read_operation("q2").numpy()
        f8_values = reader.read_operation("f8").numpy()
    np.testing.assert_array_equal(q1_values[0, :4], [1, -1, 1, -1])
    np.testing.assert_array_equal(q2_values[0, :4], [-1, 0, 1, 2])
    np.testing.assert_array_equal(f8_values[0, :3], [0, 1, -1])
    assert math.isnan(f8_values[0, 3])


def test_gguf_q8_variants(tmp_path: Path) -> None:
    q8_1 = np.array([1.0, 0.0], dtype=np.float16).tobytes() + np.arange(32, dtype=np.int8).tobytes()
    q8_k = (
        np.array([2.0], dtype=np.float32).tobytes()
        + np.arange(256, dtype=np.int8).tobytes()
        + bytes(32)
    )
    path = write_gguf(
        tmp_path / "q8.gguf",
        [("q8_1", (1, 32), 9, q8_1), ("q8_k", (1, 256), 15, q8_k)],
    )
    with open_checkpoint(path) as reader:
        np.testing.assert_array_equal(reader.read_operation("q8_1").numpy()[0], np.arange(32))
        expected = np.arange(256, dtype=np.int16).astype(np.int8).astype(np.float32) * 2
        np.testing.assert_array_equal(reader.read_operation("q8_k").numpy()[0], expected)


def test_gguf_big_endian_and_column_slice(tmp_path: Path) -> None:
    values = np.arange(16, dtype=np.float32).reshape(4, 4).byteswap()
    path = write_gguf(tmp_path / "big.gguf", [("weight", (4, 4), 0, values.tobytes())], endian=">")
    with open_checkpoint(path) as reader:
        actual = reader.read_operation("weight", columns=slice(1, 3))
    torch.testing.assert_close(actual, torch.arange(16, dtype=torch.float32).reshape(4, 4)[1:3])


def test_gguf_shards_are_combined_and_duplicate_names_fail(tmp_path: Path) -> None:
    first = write_gguf(
        tmp_path / "model-00001-of-00002.gguf",
        [("first", (1, 4), 0, np.arange(4, dtype=np.float32).tobytes())],
        metadata={"split.no": 0, "split.count": 2, "split.tensors.count": 2},
    )
    write_gguf(
        tmp_path / "model-00002-of-00002.gguf",
        [("second", (1, 4), 0, np.arange(4, dtype=np.float32).tobytes())],
        metadata={"split.no": 1, "split.count": 2, "split.tensors.count": 2},
    )
    with open_checkpoint(first) as reader:
        assert reader.tensor_names() == ("first", "second")
        assert len(reader.fingerprint) == 64
    write_gguf(
        tmp_path / "model-00002-of-00002.gguf",
        [("first", (1, 4), 0, np.arange(4, dtype=np.float32).tobytes())],
        metadata={"split.no": 1, "split.count": 2, "split.tensors.count": 2},
    )
    with pytest.raises(CapabilityError, match="duplicate tensor"):
        open_checkpoint(first)


def test_gguf_metadata_and_embedded_tokenizer(tmp_path: Path) -> None:
    tokenizer = '{"version":"1.0","model":{"type":"BPE","vocab":{"hello":0},"merges":[]}}'
    path = write_gguf(
        tmp_path / "metadata.gguf",
        [("weight", (1, 4), 0, np.zeros(4, dtype=np.float32).tobytes())],
        metadata={
            "tokenizer.huggingface.json": tokenizer,
            "tokenizer.chat_template": "{{ messages }}",
            "tokenizer.ggml.tokens": ["hello", "world"],
            "test.integer_array": [1, 2, 3],
            "tokenizer.ggml.eos_token_id": 2,
            "llama.rope.dimension_count": 4,
            "llama.expert_count": 3,
            "llama.expert_used_count": 2,
        },
    )
    with open_checkpoint(path) as reader:
        config = gguf_to_transformers_config(reader.metadata)
        assert reader.metadata["tokenizer.ggml.tokens"] == ["hello", "world"]
        assert reader.metadata["test.integer_array"] == [1, 2, 3]
        destination = materialize_client_tokenizer(reader, tmp_path / "client")
    assert config["model_type"] == "llama"
    assert config["hidden_size"] == 4
    assert config["num_attention_heads"] == 2
    assert config["num_key_value_heads"] == 1
    assert config["head_dim"] == 2
    assert config["rope_dimension_count"] == 4
    assert config["num_experts"] == 3
    assert config["num_experts_per_tok"] == 2
    assert config["rope_theta"] == 10000.0
    assert (destination / "tokenizer.json").read_text() == tokenizer
    assert "{{ messages }}" in (destination / "tokenizer_config.json").read_text()


def test_gguf_version_one_container(tmp_path: Path) -> None:
    expected = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    path = write_gguf(
        tmp_path / "v1.gguf",
        [("weight", (1, 4), 0, expected.numpy().tobytes())],
        version=1,
    )
    with open_checkpoint(path) as reader:
        torch.testing.assert_close(reader.read_operation("weight"), expected)


def _big_endian_payload(kind: int, payload: bytes) -> bytes:
    layouts = {
        2: ((0, 2, 2),),
        3: ((0, 4, 2),),
        6: ((0, 2, 2),),
        7: ((0, 4, 2),),
        8: ((0, 2, 2),),
        9: ((0, 4, 2),),
        10: ((80, 84, 2),),
        11: ((108, 110, 2),),
        12: ((0, 4, 2),),
        13: ((0, 4, 2),),
        14: ((208, 210, 2),),
        15: ((0, 4, 4), (260, 292, 2)),
        16: ((0, 2, 2), (2, 66, 4)),
        17: ((0, 2, 2), (2, 66, 2)),
        18: ((0, 2, 2), (2, 98, 4)),
        19: ((0, 2, 2), (34, 50, 2)),
        20: ((0, 2, 2),),
        21: ((0, 2, 2),),
        22: ((0, 2, 2),),
        23: ((0, 4, 2),),
        29: ((48, 56, 2),),
        30: ((0, 2, 2),),
        34: ((52, 54, 2),),
        35: ((64, 66, 2),),
        41: ((0, 2, 2),),
        42: ((0, 2, 2),),
    }
    _, width = gguf.GGML_QUANT_SIZES.get(kind, (1, len(payload)))
    rows = len(payload) // width
    result = np.frombuffer(payload, dtype=np.uint8).copy().reshape(rows, width)
    for start, stop, word in layouts.get(kind, ()):
        dtype = np.dtype(f"u{word}")
        for row in result:
            row[start:stop] = row[start:stop].view(dtype).byteswap().view(np.uint8)
    if kind == 0:
        result = result.view(np.float32).byteswap().view(np.uint8)
    elif kind == 1:
        result = result.view(np.float16).byteswap().view(np.uint8)
    return result.tobytes()


@pytest.mark.parametrize("kind", SUPPORTED_BY_UPSTREAM, ids=lambda kind: kind.name)
def test_gguf_big_endian_quantized_matches_little_endian_reference(
    kind: gguf.GGMLQuantizationType, tmp_path: Path
) -> None:
    block, width = gguf.GGML_QUANT_SIZES[kind]
    logical_shape = (2, block)
    source = np.linspace(-1, 1, num=np.prod(logical_shape), dtype=np.float32).reshape(logical_shape)
    try:
        payload = gguf.quantize(source, kind).tobytes()
    except NotImplementedError:
        payload = (
            np.random.default_rng(kind.value)
            .integers(0, 256, size=np.prod(logical_shape) // block * width, dtype=np.uint8)
            .tobytes()
        )
    big_payload = _big_endian_payload(kind.value, payload)
    path = write_gguf(
        tmp_path / "big-quant.gguf",
        [("weight", logical_shape, kind.value, big_payload)],
        endian=">",
    )
    with open_checkpoint(path) as reader:
        actual = reader.read_operation("weight").numpy()
    expected = gguf.dequantize(np.frombuffer(payload, dtype=np.uint8).reshape(2, width), kind)
    np.testing.assert_array_equal(actual, expected)


def test_gguf_big_endian_q1_q2_f8(tmp_path: Path) -> None:
    q1 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b01010101] * 16)
    q2 = np.array([1.0], dtype=np.float16).tobytes() + bytes([0b11100100] * 16)
    f8 = bytes([0x00, 0x38, 0xB8, 0x7F])
    path = write_gguf(
        tmp_path / "big-new.gguf",
        [
            ("q1", (1, 128), 41, _big_endian_payload(41, q1)),
            ("q2", (1, 64), 42, _big_endian_payload(42, q2)),
            ("f8", (1, 4), 43, f8),
        ],
        endian=">",
    )
    with open_checkpoint(path) as reader:
        q1_values = reader.read_operation("q1").numpy()
        q2_values = reader.read_operation("q2").numpy()
        f8_values = reader.read_operation("f8").numpy()
    np.testing.assert_array_equal(q1_values[0, :4], [1, -1, 1, -1])
    np.testing.assert_array_equal(q2_values[0, :4], [-1, 0, 1, 2])
    np.testing.assert_array_equal(f8_values[0, :3], [0, 1, -1])
    assert math.isnan(f8_values[0, 3])
