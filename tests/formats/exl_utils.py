from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def pack_columns(values: torch.Tensor, bits: int) -> torch.Tensor:
    rows, columns = values.shape
    output = torch.zeros(((rows * bits + 31) // 32, columns), dtype=torch.int64)
    for row in range(rows):
        for column in range(columns):
            bit = row * bits
            target = bit // 32
            shift = bit % 32
            value = int(values[row, column])
            output[target, column] |= value << shift
            if shift + bits > 32:
                output[target + 1, column] |= value >> (32 - shift)
    return output.to(torch.int32)


def pack_4bit_rows(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros((values.shape[0], values.shape[1] // 8), dtype=torch.int64)
    for index in range(8):
        packed |= values[:, index::8].to(torch.int64) << (index * 4)
    return packed.to(torch.int32)


def pack_scales(values: torch.Tensor) -> torch.Tensor:
    return pack_4bit_rows(values - 1)


def exl2_fixture(
    root: Path,
    *,
    permutation: bool = True,
    output_columns: int = 32,
    actual_columns: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.layers.0.self_attn.q_proj"
    input_rows = 64
    padded_columns = ((output_columns + 31) // 32) * 32
    generator = torch.Generator().manual_seed(17)
    inverse = (
        torch.randperm(input_rows, generator=generator).to(torch.int32) if permutation else None
    )
    groups = [(2, 0), (4, 2)]
    quantized_groups = []
    scales = []
    scale_max = torch.tensor([0.25, 0.5], dtype=torch.float16)
    for bits, _ in groups:
        rows = 32
        quantized_groups.append(
            torch.randint(
                0, 2**bits, (rows, padded_columns), generator=generator, dtype=torch.int16
            )
        )
        scales.append(
            torch.randint(1, 17, (padded_columns,), generator=generator, dtype=torch.int16)
        )
    q_weight = torch.cat(
        [
            pack_columns(values, bits)
            for values, (bits, _) in zip(quantized_groups, groups, strict=True)
        ]
    )
    q_scale = pack_scales(torch.stack(scales))
    tensors = {
        f"{prefix}.q_weight": q_weight,
        f"{prefix}.q_scale": q_scale,
        f"{prefix}.q_scale_max": scale_max,
        f"{prefix}.q_groups": torch.tensor(groups, dtype=torch.int32).to(torch.int16).flatten(),
    }
    if inverse is not None:
        tensors[f"{prefix}.q_invperm"] = inverse
    expected_padded = torch.empty((input_rows, padded_columns), dtype=torch.float16)
    for group, (bits, _) in enumerate(groups):
        scale = ((scales[group].to(torch.float16) ** 2) * scale_max[group]).to(torch.float16)
        start = group * 32
        expected_padded[start : start + 32] = (
            quantized_groups[group].to(torch.float16) - 2 ** (bits - 1)
        ) * scale
    if inverse is not None:
        permutation_values = torch.argsort(inverse)
        expected = torch.empty_like(expected_padded)
        expected[permutation_values] = expected_padded
    else:
        expected = expected_padded
    count = actual_columns if actual_columns is not None else output_columns
    expected = expected[:, :count].T.contiguous()
    config = {
        "model_type": "llama",
        "hidden_size": input_rows,
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "max_position_embeddings": 32,
    }
    (root / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(root / "model.safetensors"))
    return tensors, expected


def gptq_fixture(root: Path) -> torch.Tensor:
    prefix = "model.layers.0.mlp.up_proj"
    input_rows, output_columns, groups = 32, 16, 2
    generator = torch.Generator().manual_seed(19)
    quantized = torch.randint(
        0, 16, (input_rows, output_columns), generator=generator, dtype=torch.int32
    )
    zeros = torch.randint(0, 16, (groups, output_columns), generator=generator, dtype=torch.int32)
    scales = torch.rand((groups, output_columns), generator=generator, dtype=torch.float16) + 0.25
    group_index = torch.arange(input_rows, dtype=torch.int32) // (input_rows // groups)
    tensors = {
        f"{prefix}.qweight": pack_columns(quantized, 4),
        f"{prefix}.qzeros": pack_4bit_rows(zeros),
        f"{prefix}.scales": scales,
        f"{prefix}.g_idx": group_index,
    }
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    save_file(tensors, str(root / "model.safetensors"))
    expected = torch.empty((input_rows, output_columns), dtype=torch.float16)
    for row in range(input_rows):
        group = int(group_index[row])
        expected[row] = (quantized[row] - zeros[group]).to(torch.float16) * scales[group]
    return expected.T.contiguous()
