from __future__ import annotations

from pathlib import Path

import torch

from xurcroh.algebra.rmsnorm import unit_rms_norm
from xurcroh.runtime.generation import TokenSampler
from xurcroh.shards.format import read_shard


class ClientBoundary:
    def __init__(self, path: str | Path, key: bytes) -> None:
        self.path = Path(path)
        self._tensors, self.metadata = read_shard(path, key)
        self.closed = False
        self.input_basis = self._tensors["basis.input"]
        self.output_basis = self._tensors["basis.output"]

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("client boundary is closed")
        embedding = self._tensors["embedding.weight"][token_ids].to(self.input_basis.dtype)
        return embedding @ self.input_basis.T

    def recover(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("client boundary is closed")
        return hidden.to(self.output_basis.dtype) @ self.output_basis

    def transition(self, hidden: torch.Tensor, stage_index: int) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("client boundary is closed")
        bases = self._tensors["basis.stages"]
        if stage_index < 0 or stage_index + 1 >= bases.shape[0]:
            raise ValueError("stage transition is out of range")
        return hidden.to(bases.dtype) @ bases[stage_index] @ bases[stage_index + 1].T

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        recovered = self.recover(hidden)
        epsilon = float(self._tensors["final_norm.epsilon"])
        normalized = unit_rms_norm(recovered, epsilon)
        normalized = normalized * self._tensors.get(
            "final_norm.weight",
            torch.ones((), dtype=normalized.dtype, device=normalized.device),
        )
        lm_head = self._tensors["lm_head.weight"]
        return normalized.to(lm_head.dtype) @ lm_head.T

    def sample(self, hidden: torch.Tensor, sampler: TokenSampler) -> torch.Tensor:
        return sampler(self.logits(hidden))

    def close(self) -> None:
        if not self.closed:
            for tensor in self._tensors.values():
                tensor.zero_()
            self._tensors.clear()
            self.closed = True
