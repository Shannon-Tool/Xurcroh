from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from xurcroh.algebra.attention import AttentionWeights


class FlowEulerScheduler:
    def __init__(self, timesteps: list[float]) -> None:
        if len(timesteps) < 2 or any(later >= earlier for earlier, later in pairwise(timesteps)):
            raise ValueError("flow scheduler timesteps must strictly decrease")
        self.timesteps = timesteps

    def step(self, latent: torch.Tensor, velocity: torch.Tensor, index: int) -> torch.Tensor:
        if index < 0 or index + 1 >= len(self.timesteps):
            raise IndexError("Euler step index is outside the schedule")
        delta = self.timesteps[index] - self.timesteps[index + 1]
        return latent - delta * velocity


@dataclass(frozen=True)
class JointAttentionWeights:
    image: AttentionWeights
    text: AttentionWeights


def reference_joint_attention(
    image: torch.Tensor,
    text: torch.Tensor,
    weights: JointAttentionWeights,
    query_heads: int,
    head_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tokens, text_tokens = image.shape[0], text.shape[0]
    image_q = (image @ weights.image.q.T).reshape(image_tokens, query_heads, head_dimension)
    image_k = (image @ weights.image.k.T).reshape(image_tokens, query_heads, head_dimension)
    image_v = (image @ weights.image.v.T).reshape(image_tokens, query_heads, head_dimension)
    text_q = (text @ weights.text.q.T).reshape(text_tokens, query_heads, head_dimension)
    text_k = (text @ weights.text.k.T).reshape(text_tokens, query_heads, head_dimension)
    text_v = (text @ weights.text.v.T).reshape(text_tokens, query_heads, head_dimension)
    keys = torch.cat([image_k, text_k], dim=0)
    values = torch.cat([image_v, text_v], dim=0)

    def attend(query: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("thd,shd->hts", query, keys) / head_dimension**0.5
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum("hts,shd->thd", probabilities, values)

    image_context = attend(image_q).reshape(image_tokens, query_heads * head_dimension)
    text_context = attend(text_q).reshape(text_tokens, query_heads * head_dimension)
    return image_context @ weights.image.o.T, text_context @ weights.text.o.T
