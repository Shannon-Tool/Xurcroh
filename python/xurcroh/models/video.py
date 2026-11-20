from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


def timestep_expert(
    timestep: float,
    *,
    threshold: float,
    prompt_hash: int | None = None,
) -> str:
    del prompt_hash
    return "high_noise" if timestep >= threshold else "low_noise"


def cache_dit_difference_norm(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, (current - previous).norm(dim=-1))


def chunked_video_decode(
    latent: torch.Tensor,
    decoder: Callable[[torch.Tensor], torch.Tensor],
    *,
    chunk_frames: int = 4,
) -> torch.Tensor:
    if latent.ndim < 2 or chunk_frames < 1:
        raise ValueError("video latent must include frame and feature dimensions")
    chunks = [
        decoder(latent[start : start + chunk_frames])
        for start in range(0, latent.shape[0], chunk_frames)
    ]
    return torch.cat(chunks, dim=0)
