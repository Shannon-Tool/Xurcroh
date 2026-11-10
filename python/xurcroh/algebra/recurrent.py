from __future__ import annotations

import torch


def rotate_state(
    state: torch.Tensor,
    value_basis: torch.Tensor,
    key_basis: torch.Tensor,
) -> torch.Tensor:
    return value_basis @ state @ key_basis.T


def delta_step(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    beta: float,
    decay: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decayed = state * decay
    update = beta * (value - decayed @ key)
    updated = decayed + torch.outer(update, key)
    return updated, updated @ query


def frame_delta_step(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 2 or values.ndim != 2 or queries.ndim != 2:
        raise ValueError("frame DeltaNet keys, values, and queries must be token matrices")
    if not (keys.shape[0] == values.shape[0] == queries.shape[0] == beta.shape[0]):
        raise ValueError("frame DeltaNet token counts must match")
    a = keys.T @ (beta[:, None] * keys)
    b = values.T @ (beta[:, None] * keys)
    identity = torch.eye(keys.shape[1], dtype=keys.dtype, device=keys.device)
    updated = torch.linalg.solve((identity + a).T, (state + b).T).T
    return updated, updated @ queries.T
