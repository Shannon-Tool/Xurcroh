from __future__ import annotations

import torch


def interleaved_frequencies(head_dimension: int, *, base: float = 10_000) -> torch.Tensor:
    if head_dimension < 2 or head_dimension % 2:
        raise ValueError("interleaved MRoPE requires a positive even head dimension")
    if base <= 1:
        raise ValueError("MRoPE base must be greater than one")
    pairs = head_dimension // 2
    bands = torch.arange(pairs, dtype=torch.float64) // 3
    return base ** (-2 * bands / pairs)


def apply_interleaved_mrope(
    values: torch.Tensor,
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-1] % 2:
        raise ValueError("MRoPE input must have token, head, and even channel dimensions")
    pairs = values.shape[-1] // 2
    if positions.shape != (values.shape[0], 3):
        raise ValueError("MRoPE positions must contain temporal, height, and width axes")
    if frequencies.shape != (pairs,):
        raise ValueError("MRoPE frequencies must match rotary pairs")
    axis_indices = torch.arange(pairs, device=positions.device) % 3
    selected_positions = positions[:, axis_indices]
    angles = selected_positions.to(dtype=frequencies.dtype) * frequencies[None, :]
    angles = angles.to(device=values.device, dtype=values.dtype)
    paired = values.reshape(values.shape[0], values.shape[1], pairs, 2)
    x, y = paired[..., 0], paired[..., 1]
    cosine, sine = torch.cos(angles)[:, None], torch.sin(angles)[:, None]
    rotated = torch.stack([x * cosine - y * sine, x * sine + y * cosine], dim=-1)
    return rotated.reshape_as(values)
