from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from safetensors import safe_open

from xurcroh.errors import ConfigurationError, ModelLoadError
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.model_compiler import _compile_matrix
from xurcroh.fortress.placement import Operation


def discover_diffusers_operations(
    checkpoint: str | Path, *, component: str = "transformer"
) -> list[Operation]:
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ConfigurationError("diffusers component must be a single directory name")
    component_root = Path(checkpoint) / component
    if not component_root.is_dir():
        raise ModelLoadError(f"diffusers component directory is unavailable: {component_root}")
    files = sorted(component_root.rglob("*.safetensors"))
    if not files:
        raise ModelLoadError(f"no safetensors weights found in {component_root}")
    discovered: list[Operation] = []
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for tensor_name in handle.keys():  # noqa: SIM118
                if not tensor_name.endswith(".weight"):
                    continue
                if len(handle.get_slice(tensor_name).get_shape()) != 2:
                    continue
                name = tensor_name.removesuffix(".weight")
                kind = "attention" if ".attn." in name else "projection"
                paired_with: str | None = None
                if name.endswith(".to_q"):
                    paired_with = name.removesuffix("to_q") + "to_k"
                elif name.endswith(".to_k"):
                    paired_with = name.removesuffix("to_k") + "to_q"
                elif name.endswith(".ff.net.0.proj"):
                    paired_with = name.removesuffix("ff.net.0.proj") + "ff.net.2"
                elif name.endswith(".ff.net.2"):
                    paired_with = name.removesuffix("ff.net.2") + "ff.net.0.proj"
                if paired_with is not None:
                    paired_with += ".weight"
                discovered.append(Operation(tensor_name, len(discovered), kind, paired_with))
    if not discovered:
        raise ConfigurationError(f"no linear matrix weights found in {component_root}")
    return discovered


def compile_diffusers_component(
    checkpoint: str | Path,
    *,
    component: str = "transformer",
    threshold: int = 2,
    slice_count: int = 3,
    inner_bases: bool = True,
    seed: int | None = None,
    share_count: int = 2,
    operation_sink: Callable[[str, CompiledOperation], None] | None = None,
) -> dict[str, CompiledOperation]:
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ConfigurationError("diffusers component must be a single directory name")
    root = Path(checkpoint)
    component_root = root / component
    if not root.is_dir() or not component_root.is_dir():
        raise ModelLoadError(f"diffusers component directory is unavailable: {component_root}")
    files = sorted(component_root.rglob("*.safetensors"))
    if not files:
        raise ModelLoadError(f"no safetensors weights found in {component_root}")
    operations: dict[str, CompiledOperation] = {}
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                if not name.endswith(".weight"):
                    continue
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2:
                    tensor.zero_()
                    continue
                _compile_matrix(
                    name,
                    tensor,
                    operations,
                    threshold=threshold,
                    slice_count=slice_count,
                    inner_bases=inner_bases,
                    seed=seed,
                    share_count=share_count,
                    retain_quantized=operation_sink is None,
                    operation_sink=operation_sink,
                )
                tensor.zero_()
    if not operations:
        raise ConfigurationError(f"no linear matrix weights found in {component_root}")
    return operations
