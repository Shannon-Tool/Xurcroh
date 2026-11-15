from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch

from xurcroh.config import ModelDefinition
from xurcroh.errors import CapabilityError, ConfigurationError, ModelLoadError
from xurcroh.prompting import render_chat_prompt
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.generation import (
    GenerationEvent,
    GenerationSummary,
    bounded_execution,
    check_request_health,
    normalize_stop_token_ids,
    sample_next_token,
    stopping_strings,
)


class LocalTransformersEngine:
    def __init__(self, name: str, definition: ModelDefinition, worker_id: str) -> None:
        self.name = name
        self.definition = definition
        self.worker_id = worker_id
        self.engine_name = "transformers"
        self._model: Any = None
        self._tokenizer: Any = None
        self._stop_token_ids: set[int] = set()
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False

    @property
    def modality(self) -> str:
        if self.definition.modality != "auto":
            return self.definition.modality
        path = Path(self.definition.path)
        if path.is_dir() and (path / "config.json").is_file():
            try:
                config = json.loads((path / "config.json").read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ModelLoadError(
                    f"cannot inspect model config for {self.name}: {error}"
                ) from error
            architectures = " ".join(config.get("architectures", [])).lower()
            model_type = config.get("model_type", "")
            if "diffusion" in model_type or "flux" in architectures:
                return "image"
            if "video" in model_type or "wan" in architectures or "minimax" in architectures:
                return "video"
            if "conditionalgeneration" in architectures or "vision" in config:
                return "vlm"
        return "text"

    def _load(self) -> None:
        if self._closed:
            raise ModelLoadError(f"model {self.name!r} engine is closed")
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            path = Path(self.definition.path)
            if not path.is_dir() and not self.definition.allow_download:
                raise ModelLoadError(
                    f"model {self.name!r} path {str(path)!r} is unavailable; "
                    "provide a local directory or set allow_download=true"
                )
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                options = {
                    "local_files_only": not self.definition.allow_download,
                    "trust_remote_code": False,
                    "revision": self.definition.revision,
                }
                self._tokenizer = AutoTokenizer.from_pretrained(self.definition.path, **options)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.definition.path, dtype=torch.float32, **options
                )
                self._model.eval()
                generation = getattr(self._model, "generation_config", None)
                self._stop_token_ids = normalize_stop_token_ids(
                    getattr(generation, "eos_token_id", None), self._tokenizer.eos_token_id
                )
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except Exception as error:
                self._model = None
                self._tokenizer = None
                raise ModelLoadError(f"failed to load model {self.name!r}: {error}") from error

    def generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        self._load()
        started = time.monotonic()
        with bounded_execution(self._inference_lock, cancelled, started, timeout, self.name):
            prompt = render_chat_prompt(self._tokenizer, request.messages)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            prompt_tokens = input_ids.shape[-1]
            if prompt_tokens == 0:
                raise ConfigurationError(
                    f"model {self.name!r} tokenizer produced an empty prompt; "
                    "check tokenizer compatibility and message content"
                )
            vocabulary_size = getattr(self._model.config, "vocab_size", None)
            if vocabulary_size is not None and int(input_ids.max()) >= vocabulary_size:
                raise ModelLoadError(
                    f"model {self.name!r} tokenizer emits IDs outside its "
                    f"{vocabulary_size} token vocabulary"
                )
            position_limit = getattr(self._model.config, "n_positions", None) or getattr(
                self._model.config, "max_position_embeddings", prompt_tokens + request.max_tokens
            )
            if prompt_tokens + request.max_tokens > position_limit:
                raise ConfigurationError(
                    f"prompt ({prompt_tokens} tokens) plus max_tokens ({request.max_tokens}) "
                    f"exceeds model context limit {position_limit}"
                )
            current = input_ids
            past = None
            completion_ids: list[int] = []
            text = ""
            finish_reason: Literal["stop", "length"] = "length"
            stops = stopping_strings(request)
            for _ in range(request.max_tokens):
                check_request_health(cancelled, started, timeout)
                with torch.inference_mode():
                    outputs = self._model(input_ids=current, past_key_values=past, use_cache=True)
                    next_token = sample_next_token(outputs.logits[:, -1, :], request)
                    token_id = int(next_token.item())
                    completion_ids.append(token_id)
                    updated_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
                    delta = updated_text[len(text) :]
                    text = updated_text
                    past = outputs.past_key_values
                    current = next_token
                yield GenerationEvent(delta=delta)
                if token_id in self._stop_token_ids:
                    finish_reason = "stop"
                    break
                if stops and any(text.endswith(stop) for stop in stops):
                    finish_reason = "stop"
                    break
            check_request_health(cancelled, started, timeout)
            duration_ms = (time.monotonic() - started) * 1000
            yield GenerationEvent(
                delta="",
                summary=GenerationSummary(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(completion_ids),
                    finish_reason=finish_reason,
                    duration_ms=duration_ms,
                ),
            )

    def close(self) -> None:
        self._closed = True
        if self._inference_lock.acquire(blocking=False):
            self._model = None
            self._tokenizer = None
            self._stop_token_ids.clear()
            self._inference_lock.release()

    def require_modality(self, expected: str) -> None:
        actual = self.modality
        if actual != expected:
            raise CapabilityError(
                f"model {self.name!r} does not support {expected} generation (detected {actual})"
            )
