from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from safetensors import safe_open

from xurcroh.errors import ConfigurationError, ModelLoadError
from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.model_compiler import _compile_matrix


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
