from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


class CubePermutation:
    def __init__(
        self,
        *,
        grid_shape: tuple[int, int, int],
        cube_shape: tuple[int, int, int],
        seed: int,
    ) -> None:
        if any(
            grid % cube or grid <= 0 or cube <= 0
            for grid, cube in zip(grid_shape, cube_shape, strict=True)
        ):
            raise ValueError("cube shape must divide the spatio-temporal grid")
        self.grid_shape = grid_shape
        self.cube_shape = cube_shape
        self.macro_shape = tuple(
            grid // cube for grid, cube in zip(grid_shape, cube_shape, strict=True)
        )
        generator = torch.Generator().manual_seed(seed)
        macro_count = self.macro_shape[0] * self.macro_shape[1] * self.macro_shape[2]
        self.macro_permutation = torch.randperm(macro_count, generator=generator)
        micro_count = cube_shape[0] * cube_shape[1] * cube_shape[2]
        self.micro_permutations = [
            torch.randperm(micro_count, generator=generator) for _ in range(macro_count)
        ]
        total = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.permutation = torch.empty(total, dtype=torch.long)
        for new_macro, original_macro in enumerate(self.macro_permutation.tolist()):
            source = self.cube_indices(original_macro)
            destination = self.cube_indices(new_macro)
            micro = self.micro_permutations[new_macro]
            self.permutation[destination] = source[micro]
        if self.permutation.numel() != grid_shape[0] * grid_shape[1] * grid_shape[2]:
            raise ValueError("cube permutation does not cover the grid")

    def cube_indices(self, macro: int) -> torch.Tensor:
        _mt, mh, mw = self.macro_shape
        ct, ch, cw = self.cube_shape
        temporal, remainder = divmod(macro, mh * mw)
        height, width = divmod(remainder, mw)
        positions = []
        for t in range(ct):
            for h in range(ch):
                for w in range(cw):
                    positions.append(
                        ((temporal * ct + t) * self.grid_shape[1] + height * ch + h)
                        * self.grid_shape[2]
                        + width * cw
                        + w
                    )
        return torch.tensor(positions, dtype=torch.long)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return values[self.permutation.to(values.device)]

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values[torch.argsort(self.permutation).to(values.device)]


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
