from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from xurcroh.errors import ModelLoadError

if TYPE_CHECKING:
    from xurcroh.formats import TensorReader


def _write_private(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)


def materialize_client_tokenizer(reader: TensorReader, trusted_dir: str | Path) -> Path:
    reader._ensure_open()
    serialized = reader.metadata.get("tokenizer.huggingface.json")
    if not isinstance(serialized, str):
        raise ModelLoadError("GGUF checkpoint has no embedded Hugging Face tokenizer JSON")
    try:
        parsed = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ModelLoadError("embedded Hugging Face tokenizer is invalid JSON") from error
    if not isinstance(parsed, dict):
        raise ModelLoadError("embedded Hugging Face tokenizer must be a JSON object")
    destination = Path(trusted_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_private(destination / "tokenizer.json", serialized)
    configuration = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": reader.metadata.get("tokenizer.chat_template"),
    }
    for public, gguf_key in (
        ("bos_token_id", "tokenizer.ggml.bos_token_id"),
        ("eos_token_id", "tokenizer.ggml.eos_token_id"),
        ("unk_token_id", "tokenizer.ggml.unknown_token_id"),
        ("pad_token_id", "tokenizer.ggml.padding_token_id"),
        ("sep_token_id", "tokenizer.ggml.separator_token_id"),
    ):
        if gguf_key in reader.metadata:
            configuration[public] = reader.metadata[gguf_key]
    _write_private(
        destination / "tokenizer_config.json", json.dumps(configuration, sort_keys=True) + "\n"
    )
    return destination
