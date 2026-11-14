from __future__ import annotations

from typing import Any

import torch

from xurcroh.fortress.compiler import CompiledOperation
from xurcroh.fortress.text_loader import CLIENT_ONLY


def load_multimodal_decoder(
    checkpoint: str,
    operations: dict[str, CompiledOperation],
    dtype: torch.dtype,
) -> tuple[Any, dict[str, CompiledOperation]]:
    from transformers import AutoModelForMultimodalLM

    from xurcroh.fortress.module import replace_linear_layers

    model = AutoModelForMultimodalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )
    client_towers = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not name.startswith("model.language_model.")
    )
    replace_linear_layers(
        model,
        operations,
        client_only=(*CLIENT_ONLY, *client_towers),
    )
    return model, operations
