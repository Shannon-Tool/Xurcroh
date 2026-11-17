from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from xurcroh.config import ModelDefinition
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.diffusers_loader import (
    compile_diffusers_component,
    discover_diffusers_operations,
)
from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
from xurcroh.fortress.placement import Operation


def _namespaced(component: str, operation: Operation) -> Operation:
    paired = operation.paired_with
    if isinstance(paired, str):
        paired = f"{component}:{paired}"
    elif isinstance(paired, tuple):
        paired = tuple(f"{component}:{name}" for name in paired)
    return replace(operation, identifier=f"{component}:{operation.identifier}", paired_with=paired)


def discover_media_operations(
    checkpoint: str, adapter: MediaArchitectureAdapter
) -> list[Operation]:
    discovered = []
    for component in adapter.components:
        for operation in discover_diffusers_operations(checkpoint, component=component):
            discovered.append(_namespaced(component, operation))
    return discovered


def compile_media_checkpoint(
    definition: ModelDefinition,
    adapter: MediaArchitectureAdapter,
    *,
    operation_sink: Callable[[str, CompiledOperation], None] | None = None,
) -> dict[str, dict[str, CompiledOperation]]:
    compiled = {}
    for component in adapter.components:
        sink = operation_sink
        if operation_sink is not None:

            def sink(name: str, operation: CompiledOperation, _component: str = component) -> None:
                operation_sink(f"{_component}:{name}", operation)

        compiled[component] = compile_diffusers_component(
            definition.path,
            component=component,
            seed=definition.fortress_seed,
            share_count=definition.fortress_shares,
            operation_sink=sink,
        )
    return compiled
