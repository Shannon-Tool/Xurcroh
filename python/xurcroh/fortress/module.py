from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

import torch

from xurcroh.fortress.arithmetic import (
    activation_row_seed,
    quantize_activations,
    seeded_split_secret,
    signed_field,
    split_secret,
)
from xurcroh.fortress.compiler import (
    CompiledOperation,
    apply_input_basis,
    apply_output_basis,
)
from xurcroh.models.moe import ConstantLoadDispatcher, constant_load_plan

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
        reference_mode: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # The weight lives in the grid, so this module would otherwise have no
        # parameters. Host graphs read a submodule's dtype from its first
        # parameter (the Wan time embedder does); an empty parameter keeps
        # that query, and ``.to(dtype)``, working without holding any weight.
        self.dtype_anchor = torch.nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)
        if not 8 <= activation_bits <= 12:
            raise ValueError("Fortress activations must use 8-12 bits")
        self.name = name
        self.operation = operation
        self.grid_client = grid_client
        self.activation_bits = activation_bits
        self.bias = bias
        self.reference_mode = reference_mode

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.reference_mode:
            return self.reference(hidden)
        from xurcroh.fortress.transport import execute_native_grid

        operation = self.operation
        original_shape = hidden.shape
        flattened = hidden.reshape(-1, hidden.shape[-1])
        transformed = apply_input_basis(operation, flattened)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        activation_seed = (
            secrets.token_bytes(32)
            if self.grid_client is not None and self.grid_client.seeded_activations
            else None
        )
        if activation_seed is None:
            shares = split_secret(activation.values, count=len(operation.shares)).shares
        else:
            vector_input = activation.values.ndim == 1
            rows = activation.values.reshape(-1, activation.values.shape[-1])
            row_grids = [
                seeded_split_secret(
                    row,
                    seed=activation_row_seed(activation_seed, index),
                    count=len(operation.shares),
                ).shares
                for index, row in enumerate(rows)
            ]
            shares = [
                torch.stack([grid[index] for grid in row_grids])
                for index in range(len(operation.shares))
            ]
            if vector_input:
                shares = [share[0] for share in shares]
        product = (
            execute_native_grid(shares, operation.shares)
            if self.grid_client is None
            else self.grid_client.execute(
                shares, request_id=self.name, activation_seed=activation_seed
            )
        )
        scales = activation.scales.unsqueeze(-1)
        result = signed_field(product).to(torch.float64) * scales * operation.quantized.scale
        result = apply_output_basis(operation, result).to(hidden.dtype)
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.reshape(*original_shape[:-1], result.shape[-1])

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        operation = self.operation
        original_shape = hidden.shape
        flattened = hidden.reshape(-1, hidden.shape[-1])
        transformed = apply_input_basis(operation, flattened)
        activation = quantize_activations(transformed, bits=self.activation_bits)
        product = activation.values @ operation.quantized.require_values().to(torch.int64).T
        result = signed_field(product).to(torch.float64) * activation.scales.unsqueeze(-1)
        result = result * operation.quantized.scale
        result = apply_output_basis(operation, result).to(hidden.dtype)
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.reshape(*original_shape[:-1], result.shape[-1])


class FortressExperts(torch.nn.Module):
    gate_proj: torch.nn.ModuleList
    up_proj: torch.nn.ModuleList
    down_proj: torch.nn.ModuleList

    def __init__(
        self,
        original: Any,
        prefix: str,
        operations: dict[str, CompiledOperation],
        grid_clients: dict[str, FortressRoleClient] | None,
    ) -> None:
        super().__init__()
        expert_indices = {
            int(name[len(prefix) + 1 :].split(".", 1)[0])
            for name in operations
            if name.startswith(prefix + ".")
        }
        if not expert_indices or expert_indices != set(range(max(expert_indices) + 1)):
            raise ValueError("Fortress expert indices must be contiguous and zero-based")
        self.num_experts = len(expert_indices)
        self.act_fn = original.act_fn
        for projection in ("gate_proj", "up_proj", "down_proj"):
            modules = []
            for expert in range(self.num_experts):
                name = f"{prefix}.{expert}.{projection}"
                if name not in operations:
                    raise ValueError(f"missing Fortress expert operation: {name}")
                modules.append(
                    FortressLinear(
                        name,
                        operations[name],
                        grid_client=None if grid_clients is None else grid_clients[name],
                    )
                )
            setattr(self, projection, torch.nn.ModuleList(modules))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("Fortress expert input and routing dimensions do not match")
        top_k = top_k_index.shape[1]
        constant_load = hidden_states.shape[0] * top_k
        plan = constant_load_plan(
            top_k_index,
            top_k_weights,
            expert_count=self.num_experts,
            constant_load=constant_load,
        )
        outputs = []
        for expert in range(self.num_experts):
            selected = hidden_states[plan.token_indices[expert]]
            gate = self.gate_proj[expert](selected)
            up = self.up_proj[expert](selected)
            outputs.append(self.down_proj[expert](self.act_fn(gate) * up))
        dispatcher = ConstantLoadDispatcher(top_k=top_k, constant_load=constant_load)
        return dispatcher.combine(plan, torch.stack(outputs))


def replace_linear_layers(
    module: torch.nn.Module,
    operations: dict[str, CompiledOperation],
    *,
    grid_clients: dict[str, FortressRoleClient] | None = None,
    prefix: str = "",
    client_only: tuple[str, ...] = (),
) -> torch.nn.Module:
    for name, child in module.named_children():
        qualified = f"{prefix}{name}"
        if type(child).__name__ in {
            "DeepseekV3Experts",
            "Qwen3MoeExperts",
            "Gemma4TextExperts",
        }:
            setattr(
                module,
                name,
                FortressExperts(child, qualified, operations, grid_clients),
            )
        elif isinstance(child, torch.nn.Linear):
            operation_name = next(
                (
                    candidate
                    for candidate in (f"{qualified}.weight", qualified)
                    if candidate in operations
                ),
                qualified,
            )
            operation = operations.get(operation_name)
            if operation is None:
                if qualified in client_only or any(
                    qualified.endswith(f".{name}") for name in client_only
                ):
                    continue
                raise ValueError(f"missing Fortress operation for {qualified}.weight")
            bias = child.bias.detach().clone() if child.bias is not None else None
            replacement = FortressLinear(
                operation_name,
                operation,
                grid_client=None if grid_clients is None else grid_clients[operation_name],
                bias=bias,
                dtype=child.weight.dtype,
            )
            setattr(module, name, replacement)
        else:
            replace_linear_layers(
                child,
                operations,
                grid_clients=grid_clients,
                prefix=f"{qualified}.",
                client_only=client_only,
            )
    return module


def connect_grid_clients(
    module: torch.nn.Module,
    grid_clients: dict[str, FortressRoleClient],
) -> None:
    for child in module.modules():
        if isinstance(child, FortressLinear):
            if child.name not in grid_clients:
                raise ValueError(f"missing grid client for {child.name}")
            child.grid_client = grid_clients[child.name]


def set_reference_mode(module: torch.nn.Module, enabled: bool = True) -> None:
    for child in module.modules():
        if isinstance(child, FortressLinear):
            child.reference_mode = enabled
