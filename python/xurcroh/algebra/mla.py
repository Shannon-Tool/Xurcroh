from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MLAWeights:
    dkv: torch.Tensor
    dq: torch.Tensor
    q_abs: torch.Tensor
    o_abs: torch.Tensor
    qr: torch.Tensor
    kr: torch.Tensor


def reference_mla(hidden: torch.Tensor, weights: MLAWeights) -> tuple[torch.Tensor, torch.Tensor]:
    cache = hidden @ weights.dkv.T
    query_content = (hidden @ weights.dq.T) @ weights.q_abs.T
    query_rope = (hidden @ weights.dq.T) @ weights.qr.T
    key_rope = hidden @ weights.kr.T
    scores = query_content @ cache.T / cache.shape[-1] ** 0.5
    scores = scores + query_rope @ key_rope.T / key_rope.shape[-1] ** 0.5
    output = cache @ weights.o_abs.T
    return scores, output


def conjugate_mla(
    weights: MLAWeights,
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    latent_basis: torch.Tensor,
    query_transform: torch.Tensor,
    rope_query_transform: torch.Tensor,
    rope_key_transform: torch.Tensor,
) -> MLAWeights:
    query_inverse = torch.linalg.inv(query_transform)
    return MLAWeights(
        dkv=latent_basis @ weights.dkv @ input_basis.T,
        dq=query_transform @ weights.dq @ input_basis.T,
        q_abs=latent_basis @ weights.q_abs @ query_inverse,
        o_abs=output_basis @ weights.o_abs @ latent_basis.T,
        qr=rope_query_transform @ weights.qr @ query_inverse,
        kr=rope_key_transform @ weights.kr @ input_basis.T,
    )
