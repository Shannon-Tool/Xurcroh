from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from xurcroh.fortress.arithmetic import quantize_activations, signed_ring, split_secret
from xurcroh.fortress.compiler import (
    CompiledOperation,
    apply_input_basis,
    apply_output_basis,
)

if TYPE_CHECKING:
    from xurcroh.fortress.transport import FortressRoleClient


class FortressLinear(torch.nn.Module):
    def __init__(
        self,
        name: str,
        operation: CompiledOperation,
        *,
        grid_client: FortressRoleClient | None = None,
        bias: torch.Tensor | None = None,
        activation_bits: int = 12,
    ) -> None:
        super().__init__()
        if not 8 <= activation_bits <= 12:
            raise ValueError("Fortress activations must use 8-12 bits")
        self.name = name
        self.operation = operation
        self.grid_client = grid_client
        self.activation_bits = activation_bits
        self.bias = bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        from xurcroh.fortress.transport import execute_native_grid

        operation = self.operation
        original_shape = hidden.shape
        flattened = hidden.reshape(-1, hidden.shape[-1])
        transformed = apply_input_basis(operation, flattened)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        shares = split_secret(activation.values, count=len(operation.shares)).shares
        product = (
            execute_native_grid(shares, operation.shares)
            if self.grid_client is None
            else self.grid_client.execute(shares, request_id=self.name)
        )
        scales = activation.scales.unsqueeze(-1)
        result = signed_ring(product).to(torch.float64) * scales * operation.quantized.scale
        result = apply_output_basis(operation, result).to(hidden.dtype)
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.reshape(*original_shape[:-1], result.shape[-1])

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        reference = FortressLinear(
            self.name,
            self.operation,
            bias=self.bias,
            activation_bits=self.activation_bits,
        )
        return cast(torch.Tensor, reference(hidden))


def replace_linear_layers(
    module: torch.nn.Module,
    operations: dict[str, CompiledOperation],
    *,
    grid_clients: dict[str, FortressRoleClient] | None = None,
    prefix: str = "",
) -> torch.nn.Module:
    for name, child in module.named_children():
        qualified = f"{prefix}{name}"
        if isinstance(child, torch.nn.Linear):
            operation = operations.get(f"{qualified}.weight")
            if operation is None:
                raise ValueError(f"missing Fortress operation for {qualified}.weight")
            bias = child.bias.detach().clone() if child.bias is not None else None
            replacement = FortressLinear(
                f"{qualified}.weight",
                operation,
                grid_client=None if grid_clients is None else grid_clients[f"{qualified}.weight"],
                bias=bias,
            )
            setattr(module, name, replacement)
        else:
            replace_linear_layers(
                child, operations, grid_clients=grid_clients, prefix=f"{qualified}."
            )
    return module
