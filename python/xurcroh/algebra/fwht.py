from __future__ import annotations

import math

import torch


def _validate_dimension(dimension: int) -> None:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("FWHT dimension must be a power of two; padding changes model semantics")


def hadamard_matrix(
    dimension: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_dimension(dimension)
    matrix = torch.ones(1, 1, dtype=dtype, device=device)
    while matrix.shape[0] < dimension:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix


def fwht(
    values: torch.Tensor,
    signs: torch.Tensor | None = None,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if values.ndim < 1:
        raise ValueError("FWHT input must have a transform dimension")
    dimension = values.shape[-1]
    _validate_dimension(dimension)
    if signs is not None:
        if signs.shape != (dimension,):
            raise ValueError("FWHT signs must match the transform dimension")
        if not torch.all((signs == 1) | (signs == -1)):
            raise ValueError("FWHT signs must contain only +1 or -1")
        signs = signs.to(device=values.device, dtype=values.dtype)
        values = values * signs if not inverse else values

    result = values.clone()
    block = 1
    while block < dimension:
        reshaped = result.reshape(*result.shape[:-1], dimension // (2 * block), 2, block)
        left = reshaped[..., 0, :].clone()
        right = reshaped[..., 1, :].clone()
        reshaped[..., 0, :] = left + right
        reshaped[..., 1, :] = left - right
        block *= 2
    result = result / math.sqrt(dimension)
    if inverse and signs is not None:
        result = result * signs
    return result
