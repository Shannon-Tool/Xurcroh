from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

import torch

from xurcroh.algebra.attention import (
    AttentionWeights,
    reference_attention,
    reference_attention_cached,
)
from xurcroh.algebra.feedforward import FeedForwardWeights, reference_feedforward
from xurcroh.algebra.rmsnorm import unit_rms_norm
from xurcroh.models.moe import MoELayerWeights, execute_moe_layer, execute_moe_layer_cached


@dataclass(frozen=True)
class DenseLayerWeights:
    attention: AttentionWeights
    feedforward: FeedForwardWeights
    input_gain: torch.Tensor
    post_attention_gain: torch.Tensor
    epsilon: float = 1e-6


def layer_weights_from_dict(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    *,
    original_weights: bool = False,
) -> DenseLayerWeights:
    dtype = tensors[f"{prefix}.self_attn.q_proj.weight"].dtype
    if original_weights:
        input_gain = tensors[f"{prefix}.input_layernorm.weight"].to(dtype)
        post_gain = tensors[f"{prefix}.post_attention_layernorm.weight"].to(dtype)
    else:
        input_gain = torch.ones_like(tensors[f"{prefix}.input_layernorm.weight"], dtype=dtype)
        post_gain = torch.ones_like(
            tensors[f"{prefix}.post_attention_layernorm.weight"], dtype=dtype
        )
    return DenseLayerWeights(
        attention=AttentionWeights(
            q=tensors[f"{prefix}.self_attn.q_proj.weight"],
            k=tensors[f"{prefix}.self_attn.k_proj.weight"],
            v=tensors[f"{prefix}.self_attn.v_proj.weight"],
            o=tensors[f"{prefix}.self_attn.o_proj.weight"],
        ),
        feedforward=FeedForwardWeights(
            gate=tensors[f"{prefix}.mlp.gate_proj.weight"],
            up=tensors[f"{prefix}.mlp.up_proj.weight"],
            down=tensors[f"{prefix}.mlp.down_proj.weight"],
        ),
        input_gain=input_gain,
        post_attention_gain=post_gain,
    )


def execute_dense_layer(
    hidden: torch.Tensor,
    weights: DenseLayerWeights,
    *,
    query_heads: int,
    kv_heads: int,
    activation: Literal["silu", "gelu"] = "silu",
    rotary: bool = True,
    interleaved_rope: bool = False,
) -> torch.Tensor:
    attention_input = unit_rms_norm(hidden, weights.epsilon) * weights.input_gain
    attention = reference_attention(
        attention_input,
        weights.attention,
        query_heads,
        kv_heads,
        rotary=rotary,
        interleaved=interleaved_rope,
    )
    residual = hidden + attention
    ffn_input = unit_rms_norm(residual, weights.epsilon) * weights.post_attention_gain
    return residual + reference_feedforward(ffn_input, weights.feedforward, activation)


def execute_dense_layer_cached(
    hidden: torch.Tensor,
    weights: DenseLayerWeights,
    *,
    query_heads: int,
    kv_heads: int,
    past: tuple[torch.Tensor, torch.Tensor] | None = None,
    activation: Literal["silu", "gelu"] = "silu",
    rotary: bool = True,
    interleaved_rope: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    attention_input = unit_rms_norm(hidden, weights.epsilon) * weights.input_gain
    attention, cache = reference_attention_cached(
        attention_input,
        weights.attention,
        query_heads,
        kv_heads,
        past=past,
        rotary=rotary,
        interleaved=interleaved_rope,
    )
    residual = hidden + attention
    ffn_input = unit_rms_norm(residual, weights.epsilon) * weights.post_attention_gain
    return residual + reference_feedforward(ffn_input, weights.feedforward, activation), cache


class DenseStageRunner:
    def __init__(
        self,
        stages: list[list[DenseLayerWeights | MoELayerWeights]],
        *,
        query_heads: int = 4,
        kv_heads: int = 2,
        interleaved_rope: bool = False,
    ) -> None:
        if not stages:
            raise ValueError("dense pipeline requires at least one stage")
        self.stages = stages
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.interleaved_rope = interleaved_rope
        self._caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._cache_lock = threading.Lock()

    @property
    def cache_entries(self) -> int:
        with self._cache_lock:
            return len(self._caches)

    def drop_request(self, request_id: str) -> None:
        with self._cache_lock:
            cache = self._caches.pop(request_id, None)
        if cache is not None:
            for key, value in cache:
                key.zero_()
                value.zero_()

    def run_cached(self, request_id: str, hidden: torch.Tensor) -> torch.Tensor:
        with self._cache_lock:
            previous = self._caches.get(request_id, [])
            if previous and hidden.shape[0] != 1:
                raise ValueError("cached decode must contain exactly one new token")
            current: list[tuple[torch.Tensor, torch.Tensor]] = []
            layer_index = 0
            for layers in self.stages:
                for layer in layers:
                    past = previous[layer_index] if layer_index < len(previous) else None
                    if isinstance(layer, MoELayerWeights):
                        hidden, cache = execute_moe_layer_cached(
                            hidden,
                            layer,
                            query_heads=self.query_heads,
                            kv_heads=self.kv_heads,
                            past=past,
                            interleaved_rope=self.interleaved_rope,
                        )
                    else:
                        hidden, cache = execute_dense_layer_cached(
                            hidden,
                            layer,
                            query_heads=self.query_heads,
                            kv_heads=self.kv_heads,
                            past=past,
                            interleaved_rope=self.interleaved_rope,
                        )
                    current.append(cache)
                    layer_index += 1
            self._caches[request_id] = current
            return hidden

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        for layers in self.stages:
            for layer in layers:
                if isinstance(layer, MoELayerWeights):
                    hidden = execute_moe_layer(
                        hidden,
                        layer,
                        query_heads=self.query_heads,
                        kv_heads=self.kv_heads,
                        interleaved_rope=self.interleaved_rope,
                    )
                else:
                    hidden = execute_dense_layer(
                        hidden,
                        layer,
                        query_heads=self.query_heads,
                        kv_heads=self.kv_heads,
                        interleaved_rope=self.interleaved_rope,
                    )
        return hidden
