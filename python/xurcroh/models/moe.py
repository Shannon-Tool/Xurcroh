from __future__ import annotations

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


@dataclass(frozen=True)
class DispatchPlan:
    token_indices: torch.Tensor
    weights: torch.Tensor
    dummy_mask: torch.Tensor
    loads: torch.Tensor
    top_k: int


def constant_load_plan(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    *,
    expert_count: int,
    constant_load: int,
) -> DispatchPlan:
    if (
        top_indices.ndim != 2
        or top_weights.shape != top_indices.shape
        or constant_load < 1
        or expert_count < 1
    ):
        raise ValueError("invalid expert routing or constant-load dimensions")
    top_k = top_indices.shape[1]
    token_indices = torch.zeros(
        expert_count, constant_load, dtype=torch.long, device=top_indices.device
    )
    weights = torch.zeros(
        expert_count, constant_load, dtype=top_weights.dtype, device=top_weights.device
    )
    dummy_mask = torch.ones(
        expert_count, constant_load, dtype=torch.bool, device=top_indices.device
    )
    for expert in range(expert_count):
        locations = (top_indices == expert).nonzero()
        count = locations.shape[0]
        if count > constant_load:
            raise ValueError("constant-load capacity is below an expert's dropless load")
        if count:
            token_indices[expert, :count] = locations[:, 0]
            weights[expert, :count] = top_weights[locations[:, 0], locations[:, 1]]
            dummy_mask[expert, :count] = False
    loads = torch.full((expert_count,), constant_load, dtype=torch.long, device=top_indices.device)
    return DispatchPlan(token_indices, weights, dummy_mask, loads, top_k)


class ConstantLoadDispatcher:
    def __init__(self, *, top_k: int, constant_load: int) -> None:
        if top_k < 1 or constant_load < 1:
            raise ValueError("top_k and constant_load must be positive")
        self.top_k = top_k
        self.constant_load = constant_load

    def dispatch(self, hidden: torch.Tensor, router: torch.Tensor) -> DispatchPlan:
        if hidden.ndim != 2 or router.ndim != 2 or hidden.shape[1] != router.shape[1]:
            raise ValueError("hidden and router dimensions do not match")
        if self.top_k > router.shape[0]:
            raise ValueError("top_k exceeds expert count")
        logits = hidden @ router.T
        top_weights, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_weights, dim=-1)
        return constant_load_plan(
            top_indices,
            top_weights,
            expert_count=router.shape[0],
            constant_load=self.constant_load,
        )

    def combine(self, plan: DispatchPlan, expert_outputs: torch.Tensor) -> torch.Tensor:
        if expert_outputs.shape[:2] != plan.weights.shape:
            raise ValueError("expert output shape does not match constant-load plan")
        max_token = int(plan.token_indices.max().item())
        combined = expert_outputs.new_zeros(max_token + 1, expert_outputs.shape[2])
        for expert in range(expert_outputs.shape[0]):
            real = ~plan.dummy_mask[expert]
            indices = plan.token_indices[expert, real]
            contributions = expert_outputs[expert, real] * plan.weights[expert, real, None]
            combined.index_add_(0, indices, contributions)
        return combined


def reference_moe(
    hidden: torch.Tensor,
    router: torch.Tensor,
    expert_outputs: torch.Tensor,
    plan: DispatchPlan,
    *,
    top_k: int,
) -> torch.Tensor:
    logits = hidden @ router.T
    weights, indices = torch.topk(logits, top_k, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    output = torch.zeros(hidden.shape[0], expert_outputs.shape[2], dtype=expert_outputs.dtype)
    for expert in range(router.shape[0]):
        locations = (indices == expert).nonzero()
        if not locations.shape[0]:
            continue
        for location in locations:
            token_index, slot = location[0], location[1]
            plan_position = int((plan.token_indices[expert] == token_index).nonzero()[0, 0])
            output[token_index] += (
                expert_outputs[expert, plan_position] * weights[token_index, slot]
            )
    return output


@dataclass(frozen=True)
class MoELayerWeights:
    attention: AttentionWeights
    router: torch.Tensor
    experts: tuple[FeedForwardWeights, ...]
    input_gain: torch.Tensor
    post_attention_gain: torch.Tensor
    top_k: int
    epsilon: float = 1e-6


def moe_layer_weights_from_dict(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    *,
    original_weights: bool = False,
    top_k: int = 1,
) -> MoELayerWeights:
    dtype = tensors[f"{prefix}.self_attn.q_proj.weight"].dtype
    if original_weights:
        input_gain = tensors[f"{prefix}.input_layernorm.weight"].to(dtype)
        post_gain = tensors[f"{prefix}.post_attention_layernorm.weight"].to(dtype)
    else:
        input_gain = torch.ones_like(tensors[f"{prefix}.input_layernorm.weight"], dtype=dtype)
        post_gain = torch.ones_like(
            tensors[f"{prefix}.post_attention_layernorm.weight"], dtype=dtype
        )
    mlp = f"{prefix}.mlp"
    expert_indices = sorted(
        int(name.split(".experts.")[1].split(".")[0])
        for name in tensors
        if name.startswith(f"{mlp}.experts.") and name.endswith(".gate_proj.weight")
    )
    experts = tuple(
        FeedForwardWeights(
            gate=tensors[f"{mlp}.experts.{index}.gate_proj.weight"],
            up=tensors[f"{mlp}.experts.{index}.up_proj.weight"],
            down=tensors[f"{mlp}.experts.{index}.down_proj.weight"],
        )
        for index in expert_indices
    )
    if not experts:
        raise ValueError(f"MoE layer {prefix} has no experts")
    return MoELayerWeights(
        attention=AttentionWeights(
            q=tensors[f"{prefix}.self_attn.q_proj.weight"],
            k=tensors[f"{prefix}.self_attn.k_proj.weight"],
            v=tensors[f"{prefix}.self_attn.v_proj.weight"],
            o=tensors[f"{prefix}.self_attn.o_proj.weight"],
        ),
        router=tensors[f"{mlp}.gate.weight"],
        experts=experts,
        input_gain=input_gain,
        post_attention_gain=post_gain,
        top_k=top_k,
    )


def execute_moe_layer(
    hidden: torch.Tensor,
    weights: MoELayerWeights,
    *,
    query_heads: int,
    kv_heads: int,
    activation: Literal["silu", "gelu"] = "silu",
    rotary: bool = True,
    interleaved_rope: bool = False,
    rope_base: float = 10_000,
) -> torch.Tensor:
    attention_input = unit_rms_norm(hidden, weights.epsilon) * weights.input_gain
    attention = reference_attention(
        attention_input,
        weights.attention,
        query_heads,
        kv_heads,
        rotary=rotary,
        interleaved=interleaved_rope,
        norm_epsilon=weights.epsilon,
        rope_base=rope_base,
    )
    residual = hidden + attention
    ffn_input = unit_rms_norm(residual, weights.epsilon) * weights.post_attention_gain
    dispatcher = ConstantLoadDispatcher(
        top_k=weights.top_k, constant_load=ffn_input.shape[0] * weights.top_k
    )
    plan = dispatcher.dispatch(ffn_input, weights.router)
    outputs = []
    for expert, expert_weights in enumerate(weights.experts):
        selected = ffn_input[plan.token_indices[expert]]
        outputs.append(reference_feedforward(selected, expert_weights, activation))
    combined = dispatcher.combine(plan, torch.stack(outputs))
    return residual + combined


def execute_moe_layer_cached(
    hidden: torch.Tensor,
    weights: MoELayerWeights,
    *,
    query_heads: int,
    kv_heads: int,
    past: tuple[torch.Tensor, torch.Tensor] | None = None,
    activation: Literal["silu", "gelu"] = "silu",
    rotary: bool = True,
    interleaved_rope: bool = False,
    rope_base: float = 10_000,
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
        norm_epsilon=weights.epsilon,
        rope_base=rope_base,
    )
    residual = hidden + attention
    ffn_input = unit_rms_norm(residual, weights.epsilon) * weights.post_attention_gain
    dispatcher = ConstantLoadDispatcher(
        top_k=weights.top_k, constant_load=ffn_input.shape[0] * weights.top_k
    )
    plan = dispatcher.dispatch(ffn_input, weights.router)
    outputs = []
    for expert, expert_weights in enumerate(weights.experts):
        selected = ffn_input[plan.token_indices[expert]]
        outputs.append(reference_feedforward(selected, expert_weights, activation))
    combined = dispatcher.combine(plan, torch.stack(outputs))
    return residual + combined, cache
