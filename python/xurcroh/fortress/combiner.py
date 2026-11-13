from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from xurcroh.algebra.attention import apply_rope
from xurcroh.algebra.rmsnorm import unit_rms_norm
from xurcroh.fortress.arithmetic import quantize_activations, signed_ring, split_secret
from xurcroh.fortress.compiler import (
    CompiledOperation,
    apply_input_basis,
    apply_output_basis,
)
from xurcroh.fortress.grid import execute_grid
from xurcroh.models.dense import DenseLayerWeights
from xurcroh.models.moe import ConstantLoadDispatcher

if TYPE_CHECKING:
    from xurcroh.fortress.transport import FortressRoleClient


@dataclass(frozen=True)
class FortressLayerMetadata:
    head_dimension: int
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None
    input_gain: torch.Tensor
    post_attention_gain: torch.Tensor
    epsilon: float = 1e-6
    router: torch.Tensor | None = None
    top_k: int = 1


class FortressCombiner:
    def __init__(
        self,
        *,
        layers: list[DenseLayerWeights | FortressLayerMetadata],
        operations: dict[str, CompiledOperation],
        query_heads: int,
        kv_heads: int,
        final_norm: torch.Tensor,
        lm_head: torch.Tensor,
        rope_theta: float = 10_000,
        activation_bits: int = 12,
        grid_clients: dict[str, FortressRoleClient] | None = None,
    ) -> None:
        if not 8 <= activation_bits <= 12:
            raise ValueError("Fortress activations must use 8-12 bits")
        self.layers = layers
        self.operations = operations
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rope_theta = rope_theta
        self.activation_bits = activation_bits
        if grid_clients is not None and set(grid_clients) != set(operations):
            raise ValueError("every outsourced operation requires a Fortress grid client")
        self.grid_clients = grid_clients
        self._caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._lock = threading.Lock()

    @property
    def cache_entries(self) -> int:
        with self._lock:
            return len(self._caches)

    def drop_request(self, request_id: str) -> None:
        with self._lock:
            cache = self._caches.pop(request_id, None)
        if cache is not None:
            for key, value in cache:
                key.zero_()
                value.zero_()

    def _linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        operation = self.operations[name]
        transformed = apply_input_basis(operation, hidden)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        shares = split_secret(activation.values, count=len(operation.shares)).shares
        product = (
            execute_grid(shares, operation.shares)
            if self.grid_clients is None
            else self.grid_clients[name].execute(shares, request_id=name)
        )
        scales = activation.scales.unsqueeze(-1) if hidden.ndim == 2 else activation.scales
        result = signed_ring(product).to(torch.float64) * scales * operation.quantized.scale
        return apply_output_basis(operation, result)

    def _decompiled_weight(self, name: str) -> torch.Tensor:
        return self.operations[name].quantized.dequantize()

    def _attention(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        layer: DenseLayerWeights | FortressLayerMetadata,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        tokens = hidden.shape[0]
        head_dimension = (
            layer.head_dimension
            if isinstance(layer, FortressLayerMetadata)
            else layer.attention.q.shape[0] // self.query_heads
        )
        q_norm = (
            layer.q_norm if isinstance(layer, FortressLayerMetadata) else layer.attention.q_norm
        )
        k_norm = (
            layer.k_norm if isinstance(layer, FortressLayerMetadata) else layer.attention.k_norm
        )
        prefix = f"model.layers.{layer_index}.self_attn"
        q = self._linear(f"{prefix}.q_proj", hidden).reshape(
            tokens, self.query_heads, head_dimension
        )
        k = self._linear(f"{prefix}.k_proj", hidden).reshape(tokens, self.kv_heads, head_dimension)
        v = self._linear(f"{prefix}.v_proj", hidden).reshape(tokens, self.kv_heads, head_dimension)
        epsilon = layer.epsilon
        if q_norm is not None:
            q = torch.nn.functional.rms_norm(q, (head_dimension,), q_norm, epsilon)
        if k_norm is not None:
            k = torch.nn.functional.rms_norm(k, (head_dimension,), k_norm, epsilon)
        q = apply_rope(q, base=self.rope_theta, interleaved=False, position_offset=position_offset)
        k = apply_rope(k, base=self.rope_theta, interleaved=False, position_offset=position_offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=0)
            v = torch.cat([past[1], v], dim=0)
        cache = (k, v)
        groups = self.query_heads // self.kv_heads
        keys = k.repeat_interleave(groups, dim=1)
        values = v.repeat_interleave(groups, dim=1)
        scores = torch.einsum("thd,shd->hts", q, keys) / head_dimension**0.5
        total = keys.shape[0]
        causal = torch.triu(
            torch.ones(tokens, total, dtype=torch.bool, device=hidden.device),
            total - tokens + 1,
        )
        scores = scores.masked_fill(causal, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.einsum("hts,shd->thd", probabilities, values)
        context = context.reshape(tokens, self.query_heads * head_dimension)
        return self._linear(f"{prefix}.o_proj", context), cache

    def _moe_layer(
        self,
        layer_index: int,
        residual: torch.Tensor,
        layer: FortressLayerMetadata,
        *,
        reference: bool = False,
    ) -> torch.Tensor:
        if layer.router is None:
            raise ValueError("MoE layer requires a client-owned router")
        linear = self._reference_linear if reference else self._linear
        ffn_input = unit_rms_norm(residual, layer.epsilon) * layer.post_attention_gain
        dispatcher = ConstantLoadDispatcher(
            top_k=layer.top_k, constant_load=ffn_input.shape[0] * layer.top_k
        )
        plan = dispatcher.dispatch(ffn_input, layer.router)
        outputs = []
        for expert in range(layer.router.shape[0]):
            prefix = f"model.layers.{layer_index}.mlp.experts.{expert}"
            selected = ffn_input[plan.token_indices[expert]]
            gate = linear(f"{prefix}.gate_proj", selected)
            up = linear(f"{prefix}.up_proj", selected)
            activated = torch.nn.functional.silu(gate) * up
            outputs.append(linear(f"{prefix}.down_proj", activated))
        return residual + dispatcher.combine(plan, torch.stack(outputs))

    def _layer(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        layer = self.layers[layer_index]
        attention_input = unit_rms_norm(hidden, layer.epsilon) * layer.input_gain
        attention, cache = self._attention(
            layer_index,
            attention_input,
            layer,
            past=past,
            position_offset=position_offset,
        )
        residual = hidden + attention
        if isinstance(layer, FortressLayerMetadata) and layer.router is not None:
            return self._moe_layer(layer_index, residual, layer), cache
        ffn_input = unit_rms_norm(residual, layer.epsilon) * layer.post_attention_gain
        gate = self._linear(f"model.layers.{layer_index}.mlp.gate_proj", ffn_input)
        up = self._linear(f"model.layers.{layer_index}.mlp.up_proj", ffn_input)
        activated = torch.nn.functional.silu(gate) * up
        down = self._linear(f"model.layers.{layer_index}.mlp.down_proj", activated)
        return residual + down, cache

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        for index in range(len(self.layers)):
            hidden, _ = self._layer(index, hidden, position_offset=0)
        return hidden

    def run_cached(self, request_id: str, hidden: torch.Tensor) -> torch.Tensor:
        with self._lock:
            previous = self._caches.get(request_id, [])
            if previous and hidden.shape[0] != 1:
                raise ValueError("cached decode must contain exactly one token")
            current = []
            for index in range(len(self.layers)):
                past = previous[index] if index < len(previous) else None
                offset = past[0].shape[0] if past is not None else 0
                hidden, cache = self._layer(index, hidden, past=past, position_offset=offset)
                current.append(cache)
            self._caches[request_id] = current
            return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = unit_rms_norm(hidden) * self.final_norm
        return normalized @ self.lm_head.to(hidden.dtype).T

    def sample(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.logits(hidden), dim=-1)

    def _reference_linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        operation = self.operations[name]
        transformed = apply_input_basis(operation, hidden)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        product = activation.values @ operation.quantized.values.T
        scales = activation.scales.unsqueeze(-1) if hidden.ndim == 2 else activation.scales
        result = product.to(torch.float64) * scales * operation.quantized.scale
        return apply_output_basis(operation, result)

    def _reference_attention(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        layer: DenseLayerWeights | FortressLayerMetadata,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        tokens = hidden.shape[0]
        head_dimension = (
            layer.head_dimension
            if isinstance(layer, FortressLayerMetadata)
            else layer.attention.q.shape[0] // self.query_heads
        )
        q_norm = (
            layer.q_norm if isinstance(layer, FortressLayerMetadata) else layer.attention.q_norm
        )
        k_norm = (
            layer.k_norm if isinstance(layer, FortressLayerMetadata) else layer.attention.k_norm
        )
        prefix = f"model.layers.{layer_index}.self_attn"
        q = self._reference_linear(f"{prefix}.q_proj", hidden).reshape(
            tokens, self.query_heads, head_dimension
        )
        k = self._reference_linear(f"{prefix}.k_proj", hidden).reshape(
            tokens, self.kv_heads, head_dimension
        )
        v = self._reference_linear(f"{prefix}.v_proj", hidden).reshape(
            tokens, self.kv_heads, head_dimension
        )
        if q_norm is not None:
            q = torch.nn.functional.rms_norm(q, (head_dimension,), q_norm, layer.epsilon)
        if k_norm is not None:
            k = torch.nn.functional.rms_norm(k, (head_dimension,), k_norm, layer.epsilon)
        q = apply_rope(q, base=self.rope_theta, interleaved=False, position_offset=position_offset)
        k = apply_rope(k, base=self.rope_theta, interleaved=False, position_offset=position_offset)
        if past is not None:
            k = torch.cat([past[0], k], dim=0)
            v = torch.cat([past[1], v], dim=0)
        cache = (k, v)
        groups = self.query_heads // self.kv_heads
        keys = k.repeat_interleave(groups, dim=1)
        values = v.repeat_interleave(groups, dim=1)
        scores = torch.einsum("thd,shd->hts", q, keys) / head_dimension**0.5
        total = keys.shape[0]
        causal = torch.triu(
            torch.ones(tokens, total, dtype=torch.bool, device=hidden.device),
            total - tokens + 1,
        )
        scores = scores.masked_fill(causal, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.einsum("hts,shd->thd", probabilities, values)
        context = context.reshape(tokens, self.query_heads * head_dimension)
        return self._reference_linear(f"{prefix}.o_proj", context), cache

    def reference_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            attention_input = unit_rms_norm(hidden, layer.epsilon) * layer.input_gain
            attention, _ = self._reference_attention(index, attention_input, layer)
            residual = hidden + attention
            if isinstance(layer, FortressLayerMetadata) and layer.router is not None:
                hidden = self._moe_layer(index, residual, layer, reference=True)
                continue
            ffn_input = unit_rms_norm(residual, layer.epsilon) * layer.post_attention_gain
            prefix = f"model.layers.{index}.mlp"
            gate = self._reference_linear(f"{prefix}.gate_proj", ffn_input)
            up = self._reference_linear(f"{prefix}.up_proj", ffn_input)
            activated = torch.nn.functional.silu(gate) * up
            hidden = residual + self._reference_linear(f"{prefix}.down_proj", activated)
        return hidden

    def reference_sample(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = unit_rms_norm(hidden) * self.final_norm
        return torch.argmax(normalized @ self.lm_head.to(hidden.dtype).T, dim=-1)
