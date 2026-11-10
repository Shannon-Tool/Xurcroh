from __future__ import annotations

from typing import cast

import torch


def unit_rms_norm(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    mean_square = values.square().mean(dim=-1, keepdim=True)
    return values * torch.rsqrt(mean_square + epsilon)


def absorb_rms_gain(weight: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or gain.ndim != 1 or weight.shape[1] != gain.shape[0]:
        raise ValueError("RMSNorm gain must match every adjacent projection's input")
    return weight * gain.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)


def homogenize_embedding_norms(embeddings: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if embeddings.ndim < 2:
        raise ValueError("embeddings must include vocabulary and hidden dimensions")
    if epsilon <= 0:
        raise ValueError("embedding normalization epsilon must be positive")
    norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    return cast(torch.Tensor, embeddings / norms * embeddings.shape[-1] ** 0.5)
