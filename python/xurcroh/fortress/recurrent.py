from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from xurcroh.algebra.modulation import SignedPermutation
from xurcroh.algebra.recurrent import delta_step
from xurcroh.fortress.arithmetic import quantize_activations, signed_ring, split_secret
from xurcroh.fortress.compiler import (
    CompiledOperation,
    apply_input_basis,
    apply_output_basis,
    compile_operation,
)
from xurcroh.fortress.grid import execute_grid

if TYPE_CHECKING:
    from xurcroh.fortress.transport import FortressRoleClient


@dataclass(frozen=True)
class DeltaNetBlockWeights:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    beta: torch.Tensor
    decay: torch.Tensor
    output: torch.Tensor


def compile_deltanet_operations(
    weights: DeltaNetBlockWeights,
) -> dict[str, CompiledOperation]:
    operations = {}
    for name, weight in (
        ("query", weights.query),
        ("key", weights.key),
        ("value", weights.value),
        ("beta", weights.beta),
        ("decay", weights.decay),
        ("output", weights.output),
    ):
        operations[name] = compile_operation(
            weight,
            threshold=2,
            slice_count=3,
            input_basis=SignedPermutation.generate(weight.shape[1], seed=secrets.randbits(63)),
            output_basis=SignedPermutation.generate(weight.shape[0], seed=secrets.randbits(63)),
        )
    return operations


class FortressDeltaNetCombiner:
    def __init__(
        self,
        weights: DeltaNetBlockWeights,
        operations: dict[str, CompiledOperation],
        *,
        grid_clients: dict[str, FortressRoleClient] | None = None,
        activation_bits: int = 12,
    ) -> None:
        if set(operations) != {"query", "key", "value", "beta", "decay", "output"}:
            raise ValueError("DeltaNet requires all six projections")
        if grid_clients is not None and set(grid_clients) != set(operations):
            raise ValueError("every DeltaNet operation requires a grid client")
        self.weights = weights
        self.operations = operations
        self.grid_clients = grid_clients
        self.activation_bits = activation_bits

    def _linear(self, name: str, hidden: torch.Tensor) -> torch.Tensor:
        operation = self.operations[name]
        transformed = apply_input_basis(operation, hidden)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        shares = split_secret(activation.values, count=2).shares
        product = (
            execute_grid(shares, operation.shares)
            if self.grid_clients is None
            else self.grid_clients[name].execute(shares, request_id=name)
        )
        scales = activation.scales.unsqueeze(-1) if hidden.ndim == 2 else activation.scales
        result = signed_ring(product).to(torch.float64) * scales * operation.quantized.scale
        return apply_output_basis(operation, result)

    def run(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._linear("query", hidden)
        key = self._linear("key", hidden)
        value = self._linear("value", hidden)
        beta = torch.sigmoid(self._linear("beta", hidden)[..., 0])
        decay = torch.sigmoid(self._linear("decay", hidden)[..., 0])
        state = torch.zeros(key.shape[1], value.shape[1], dtype=hidden.dtype, device=hidden.device)
        outputs = []
        for index in range(hidden.shape[0]):
            state, output = delta_step(
                state,
                key[index],
                value[index],
                query[index],
                float(beta[index]),
                decay[index],
            )
            outputs.append(self._linear("output", output))
        return torch.stack(outputs), state

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        reference = FortressDeltaNetCombiner(
            self.weights, self.operations, activation_bits=self.activation_bits
        )
        return reference.run(hidden)[0]
