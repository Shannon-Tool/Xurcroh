from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from xurcroh.algebra.attention import AttentionWeights
from xurcroh.algebra.feedforward import FeedForwardWeights
from xurcroh.algebra.modulation import adaln
from xurcroh.algebra.rmsnorm import unit_rms_norm
from xurcroh.fortress.arithmetic import quantize_activations, signed_ring, split_secret
from xurcroh.fortress.compiler import CompiledOperation, compile_operation
from xurcroh.fortress.grid import execute_grid
from xurcroh.models.diffusion import FlowEulerScheduler
from xurcroh.models.dit import (
    DiTBlockWeights,
    DiTStageWeights,
    reference_dit_stage,
)

if TYPE_CHECKING:
    from xurcroh.fortress.transport import FortressRoleClient


def _attention_operations(
    operations: dict[str, CompiledOperation], prefix: str, weights: AttentionWeights
) -> None:
    for name, matrix in (
        ("q_proj", weights.q),
        ("k_proj", weights.k),
        ("v_proj", weights.v),
        ("o_proj", weights.o),
    ):
        operations[f"{prefix}.{name}"] = compile_operation(matrix, threshold=2, slice_count=3)


def _feedforward_operations(
    operations: dict[str, CompiledOperation], prefix: str, weights: FeedForwardWeights
) -> None:
    for name, matrix in (
        ("gate_proj", weights.gate),
        ("up_proj", weights.up),
        ("down_proj", weights.down),
    ):
        operations[f"{prefix}.{name}"] = compile_operation(matrix, threshold=2, slice_count=3)


def compile_dit_operations(weights: DiTStageWeights) -> dict[str, CompiledOperation]:
    operations = {
        "input": compile_operation(weights.input, threshold=2, slice_count=3),
        "output": compile_operation(weights.output, threshold=2, slice_count=3),
    }
    for index, block in enumerate(weights.blocks):
        prefix = f"blocks.{index}"
        _attention_operations(operations, f"{prefix}.image", block.image)
        _attention_operations(operations, f"{prefix}.text", block.text)
        _feedforward_operations(operations, f"{prefix}.image_ffn", block.image_ffn)
        _feedforward_operations(operations, f"{prefix}.text_ffn", block.text_ffn)
    return operations


def _quantized_attention(
    operations: dict[str, CompiledOperation], prefix: str, original: AttentionWeights
) -> AttentionWeights:
    return AttentionWeights(
        q=operations[f"{prefix}.q_proj"].quantized.dequantize(),
        k=operations[f"{prefix}.k_proj"].quantized.dequantize(),
        v=operations[f"{prefix}.v_proj"].quantized.dequantize(),
        o=operations[f"{prefix}.o_proj"].quantized.dequantize(),
        q_norm=original.q_norm,
        k_norm=original.k_norm,
    )


def _quantized_feedforward(
    operations: dict[str, CompiledOperation], prefix: str
) -> FeedForwardWeights:
    return FeedForwardWeights(
        gate=operations[f"{prefix}.gate_proj"].quantized.dequantize(),
        up=operations[f"{prefix}.up_proj"].quantized.dequantize(),
        down=operations[f"{prefix}.down_proj"].quantized.dequantize(),
    )


def quantized_dit_weights(
    original: DiTStageWeights, operations: dict[str, CompiledOperation]
) -> DiTStageWeights:
    blocks = []
    for index, block in enumerate(original.blocks):
        prefix = f"blocks.{index}"
        blocks.append(
            DiTBlockWeights(
                image=_quantized_attention(operations, f"{prefix}.image", block.image),
                text=_quantized_attention(operations, f"{prefix}.text", block.text),
                image_ffn=_quantized_feedforward(operations, f"{prefix}.image_ffn"),
                text_ffn=_quantized_feedforward(operations, f"{prefix}.text_ffn"),
                image_modulation=block.image_modulation,
                text_modulation=block.text_modulation,
                epsilon=block.epsilon,
            )
        )
    return DiTStageWeights(
        blocks=tuple(blocks),
        input=operations["input"].quantized.dequantize(),
        output=operations["output"].quantized.dequantize(),
        timesteps=original.timesteps,
        query_heads=original.query_heads,
        head_dimension=original.head_dimension,
        latent_positions=original.latent_positions,
    )


class FortressDiTCombiner:
    def __init__(
        self,
        weights: DiTStageWeights,
        operations: dict[str, CompiledOperation],
        *,
        grid_clients: dict[str, FortressRoleClient] | None = None,
        activation_bits: int = 12,
    ) -> None:
        if not 8 <= activation_bits <= 12:
            raise ValueError("Fortress activations must use 8-12 bits")
        if grid_clients is not None and set(grid_clients) != set(operations):
            raise ValueError("every outsourced DiT operation requires a Fortress grid client")
        self.weights = weights
        self.operations = operations
        self.grid_clients = grid_clients
        self.activation_bits = activation_bits

    def _linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        operation = self.operations[name]
        activation = quantize_activations(hidden, bits=self.activation_bits)
        shares = split_secret(activation.values, count=2).shares
        product = (
            execute_grid(shares, operation.shares)
            if self.grid_clients is None
            else self.grid_clients[name].execute(shares, request_id=name)
        )
        scales = activation.scales.unsqueeze(-1) if hidden.ndim == 2 else activation.scales
        return signed_ring(product).to(torch.float64) * scales * operation.quantized.scale

    def _attention(
        self, image: torch.Tensor, text: torch.Tensor, block: DiTBlockWeights, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens = image.shape[0]
        text_tokens = text.shape[0]
        heads = self.weights.query_heads
        dimension = self.weights.head_dimension
        prefix = f"blocks.{index}"

        def projections(stream: str, values: torch.Tensor, tokens: int) -> tuple[torch.Tensor, ...]:
            return tuple(
                self._linear(f"{prefix}.{stream}.{name}_proj", values).reshape(
                    tokens, heads, dimension
                )
                for name in ("q", "k", "v")
            )

        image_q, image_k, image_v = projections("image", image, image_tokens)
        text_q, text_k, text_v = projections("text", text, text_tokens)
        keys = torch.cat([image_k, text_k], dim=0)
        values = torch.cat([image_v, text_v], dim=0)

        def attend(query: torch.Tensor) -> torch.Tensor:
            scores = torch.einsum("thd,shd->hts", query, keys) / dimension**0.5
            probabilities = torch.softmax(scores, dim=-1)
            return torch.einsum("hts,shd->thd", probabilities, values)

        image_context = attend(image_q).reshape(image_tokens, heads * dimension)
        text_context = attend(text_q).reshape(text_tokens, heads * dimension)
        return (
            self._linear(f"{prefix}.image.o_proj", image_context),
            self._linear(f"{prefix}.text.o_proj", text_context),
        )

    def _feedforward(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        gate = self._linear(f"{name}.gate_proj", hidden)
        up = self._linear(f"{name}.up_proj", hidden)
        return self._linear(f"{name}.down_proj", torch.nn.functional.silu(gate) * up)

    def velocity(self, latent: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image = self._linear("input", latent)
        conditioning = text
        for index, block in enumerate(self.weights.blocks):
            image_attention, text_attention = self._attention(image, conditioning, block, index)
            image_residual = adaln(
                image,
                block.image_modulation.gamma,
                block.image_modulation.beta,
                block.image_modulation.alpha,
                image_attention,
            )
            text_residual = adaln(
                conditioning,
                block.text_modulation.gamma,
                block.text_modulation.beta,
                block.text_modulation.alpha,
                text_attention,
            )
            image = image_residual + self._feedforward(
                f"blocks.{index}.image_ffn",
                unit_rms_norm(image_residual, block.epsilon),
            )
            conditioning = text_residual + self._feedforward(
                f"blocks.{index}.text_ffn",
                unit_rms_norm(text_residual, block.epsilon),
            )
        return self._linear("output", image)

    def run(self, latent: torch.Tensor, text: torch.Tensor, *, steps: int) -> torch.Tensor:
        scheduler = FlowEulerScheduler(list(self.weights.timesteps))
        current = latent
        for index in range(steps):
            current = scheduler.step(current, self.velocity(current, text), index)
        return current

    def reference(self, latent: torch.Tensor, text: torch.Tensor, *, steps: int) -> torch.Tensor:
        reference = FortressDiTCombiner(
            self.weights, self.operations, activation_bits=self.activation_bits
        )
        return reference.run(latent, text, steps=steps)

    def original_reference(
        self, latent: torch.Tensor, text: torch.Tensor, *, steps: int
    ) -> torch.Tensor:
        return reference_dit_stage(self.weights, latent, text, steps=steps)
