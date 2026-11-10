from __future__ import annotations

from typing import cast

import torch


def orthogonal_procrustes(
    public_weight: torch.Tensor, transformed_weight: torch.Tensor
) -> torch.Tensor:
    if public_weight.shape != transformed_weight.shape:
        raise ValueError("public and transformed weights must have identical shapes")
    u, _, vh = torch.linalg.svd(public_weight.T @ transformed_weight)
    return cast(torch.Tensor, vh.T @ u.T)


def gram_spectrum(weight: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, torch.linalg.eigvalsh(weight @ weight.T).sort().values)


def nearest_embedding_by_norm(target: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    target_norms = target.norm(dim=-1, keepdim=True)
    embedding_norms = embeddings.norm(dim=-1)
    return cast(torch.Tensor, (target_norms - embedding_norms).abs().argmin(dim=-1))
