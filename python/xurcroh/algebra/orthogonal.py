from __future__ import annotations

from typing import cast

import torch


def sample_orthogonal(
    dimension: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if dimension < 1:
        raise ValueError("orthogonal dimension must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -1.0, 1.0)
    orthogonal = q * signs.unsqueeze(0)
    return cast(torch.Tensor, orthogonal.to(dtype=dtype, device=device))


def conjugate_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if weight.ndim != 2:
        raise ValueError("linear weight must be a matrix")
    if input_basis.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("input basis dimension does not match linear input")
    if output_basis.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("output basis dimension does not match linear output")
    input_basis = input_basis.to(device=weight.device, dtype=weight.dtype)
    output_basis = output_basis.to(device=weight.device, dtype=weight.dtype)
    transformed = output_basis @ weight @ input_basis.T
    transformed_bias = None if bias is None else bias @ output_basis.T
    return transformed, transformed_bias


def basis_transition(previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    if previous.shape != following.shape or previous.ndim != 2:
        raise ValueError("stage bases must be equally sized square matrices")
    if previous.shape[0] != previous.shape[1]:
        raise ValueError("stage bases must be square")
    return following @ previous.T
