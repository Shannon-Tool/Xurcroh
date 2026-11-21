from __future__ import annotations

import inspect
from typing import Any

import torch

from xurcroh.errors import ModelLoadError
from xurcroh.formats import TensorReader
from xurcroh.fortress.compiler import CompiledOperation

CLIENT_ONLY = (
    "lm_head",
    "model.per_layer_model_projection",
    "model.language_model.per_layer_model_projection",
    "per_layer_input_gate",
    "per_layer_projection",
    "vision_projection",
    "model.vision_projection",
    "multi_modal_projector",
    "audio_projection",
    "model.audio_projection",
    "model.embed_audio",
    "mlp.gate",
    "router.proj",
)


def _model_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return f"model.{name.removeprefix('model.language_model.')}"
    return name


def load_text_decoder(
    reader: TensorReader,
    operations: dict[str, CompiledOperation],
    dtype: torch.dtype,
) -> tuple[Any, dict[str, CompiledOperation]]:
    from transformers import AutoConfig, AutoModelForCausalLM

    from xurcroh.fortress.module import FortressLinear, replace_linear_layers

    text = reader.config.get("text_config", reader.config)
    if not isinstance(text, dict):
        raise ModelLoadError("multimodal checkpoint is missing text_config")
    text = dict(text)
    model_type = text.pop("model_type", f"{reader.config.get('model_type')}_text")
    config = AutoConfig.for_model(model_type, **text)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)  # type: ignore[no-untyped-call]
    model_operations = {_model_name(name): operation for name, operation in operations.items()}
    ignored = getattr(model.model, "_keys_to_ignore_on_load_unexpected", [])
    shared_weights = {f"model.{name}" for name in ignored}
    replace_linear_layers(
        model,
        model_operations,
        client_only=(*CLIENT_ONLY, *shared_weights),
    )
    model.to_empty(device="cpu")
    for name, module in list(model.named_modules()):
        if name and "RotaryEmbedding" in type(module).__name__:
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            parameters = inspect.signature(type(module)).parameters
            arguments = (
                {"config": config, "device": "cpu"}
                if "device" in parameters
                else {"config": config}
            )
            setattr(parent, child_name, type(module)(**arguments))
    # ``to_empty`` leaves derived buffers uninitialised; scaled embeddings
    # (Gemma families) restore theirs from the recorded scale.
    for module in model.modules():
        scale = getattr(module, "scalar_embed_scale", None)
        if isinstance(module, torch.nn.Embedding) and scale is not None:
            module.embed_scale = torch.tensor(float(scale))

    excluded = {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.lm_head.weight",
    }
    state: dict[str, torch.Tensor] = {}
    names = []
    for name in reader.tensor_names():
        mapped = _model_name(name)
        operation_name = mapped.removesuffix(".weight")
        if name in excluded or (
            operation_name in model_operations
            and operation_name not in CLIENT_ONLY
            and not any(operation_name.endswith(f".{part}") for part in CLIENT_ONLY)
        ):
            continue
        if mapped.startswith("model.") or mapped == "lm_head.weight":
            names.append(name)
    for name, tensor in reader.read_tensors(names).items():
        state[_model_name(name)] = tensor.to(dtype)
    # A protected projection keeps its bias on the trusted client. The bias is
    # not a registered parameter, so it is installed explicitly and a
    # checkpoint that omits it is rejected instead of running with garbage.
    for module_name, module in model.named_modules():
        if not isinstance(module, FortressLinear) or module.bias is None:
            continue
        bias = state.pop(f"{module_name}.bias", None)
        if bias is None:
            raise ModelLoadError(f"text checkpoint is missing required bias: {module_name}.bias")
        if tuple(bias.shape) != tuple(module.bias.shape):
            raise ModelLoadError(f"text checkpoint bias has the wrong shape: {module_name}.bias")
        module.bias = bias
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise ModelLoadError(
            "text checkpoint has parameters the model graph does not define: "
            + ", ".join(sorted(result.unexpected_keys)[:8])
        )
    allowed_missing = set(shared_weights)
    allowed_missing.update(
        name
        for name, module in model.named_modules()
        if isinstance(module, FortressLinear)
        for name in (f"{name}.weight", f"{name}.bias")
    )
    allowed_missing.update(("model.embed_tokens.weight", "lm_head.weight"))
    missing = set(result.missing_keys) - allowed_missing
    if missing:
        raise ModelLoadError(
            "text checkpoint is missing required parameters: " + ", ".join(sorted(missing)[:8])
        )
    return model, model_operations
