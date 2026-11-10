from __future__ import annotations

import base64
import binascii
import io
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

from xurcroh.config import ModelDefinition
from xurcroh.errors import CapabilityError, ConfigurationError, ModelLoadError
from xurcroh.requests import (
    AudioContentPart,
    CompletionRequest,
    ImageContentPart,
    InputMessage,
    TextContentPart,
)
from xurcroh.runtime.generation import GenerationEvent, GenerationSummary, check_request_health


def _decode_image(url: str) -> Image.Image:
    try:
        if url.startswith("data:image/"):
            header, encoded = url.split(",", 1)
            if not header.endswith(";base64"):
                raise ConfigurationError("image data URL must use base64 encoding")
            data = base64.b64decode(encoded, validate=True)
        elif url.startswith("file://"):
            data = Path(unquote(urlparse(url).path)).read_bytes()
        else:
            raise ConfigurationError(
                "image_url supports local file:// and base64 data URLs; remote fetching is disabled"
            )
        with Image.open(io.BytesIO(data)) as source:
            return source.convert("RGB")
    except (OSError, ValueError, binascii.Error) as error:
        raise ConfigurationError(f"invalid image input: {error}") from error


def _decode_audio(part: AudioContentPart) -> tuple[np.ndarray[Any, Any], int]:
    try:
        data = base64.b64decode(part.input_audio.data, validate=True)
        with wave.open(io.BytesIO(data), "rb") as source:
            if source.getsampwidth() != 2:
                raise ConfigurationError("WAV input must contain signed 16-bit PCM samples")
            channels = source.getnchannels()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
        if samples.size == 0:
            raise ConfigurationError("WAV input contains no samples")
        return samples.reshape(-1, channels).mean(axis=1), rate
    except (OSError, ValueError, binascii.Error, wave.Error) as error:
        raise ConfigurationError(f"invalid WAV input: {error}") from error


class LocalVisionLanguageEngine:
    def __init__(self, name: str, definition: ModelDefinition, worker_id: str) -> None:
        self.name = name
        self.definition = definition
        self.worker_id = worker_id
        self.worker_ids = [worker_id]
        self.engine_name = "transformers-vlm"
        self.modality = "vlm"
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False

    def require_modality(self, expected: str) -> None:
        if expected != "vlm":
            raise CapabilityError(f"vision-language engine does not support {expected}")

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
                raise ModelLoadError(f"vision-language model path is unavailable: {path}")
            try:
                from transformers import (
                    AutoConfig,
                    AutoModelForImageTextToText,
                    AutoProcessor,
                    AutoTokenizer,
                )

                options = {
                    "local_files_only": not self.definition.allow_download,
                    "trust_remote_code": False,
                    "revision": self.definition.revision,
                }
                config = AutoConfig.from_pretrained(self.definition.path, **options)
                tokenizer = AutoTokenizer.from_pretrained(self.definition.path, **options)
                if hasattr(config, "boi_token_index") and not hasattr(tokenizer, "boi_token_id"):
                    tokenizer.boi_token_id = config.boi_token_index
                    tokenizer.boi_token = tokenizer.convert_ids_to_tokens(config.boi_token_index)
                    tokenizer.eoi_token = tokenizer.convert_ids_to_tokens(config.eoi_token_index)
                    tokenizer.eoi_token_id = config.eoi_token_index
                    tokenizer.image_token = tokenizer.convert_ids_to_tokens(
                        config.image_token_index
                    )
                    tokenizer.image_token_id = config.image_token_index
                self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                    self.definition.path, tokenizer=tokenizer, **options
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.definition.path, dtype=torch.float32, **options
                )
                self._model.eval()
            except Exception as error:
                self._model = None
                self._processor = None
                raise ModelLoadError(f"failed to load VLM {self.name!r}: {error}") from error

    def _parts(
        self, message: InputMessage
    ) -> list[TextContentPart | ImageContentPart | AudioContentPart]:
        if isinstance(message.content, str):
            return [TextContentPart(type="text", text=message.content)]
        return message.content

    def _inputs(self, request: CompletionRequest) -> tuple[Any, int]:
        images: list[Image.Image] = []
        audio: list[np.ndarray[Any, Any]] = []
        sample_rates: list[int] = []
        template_messages = []
        fallback_messages = []
        for message in request.messages:
            template_parts = []
            fallback = ""
            for part in self._parts(message):
                if isinstance(part, TextContentPart):
                    template_parts.append({"type": "text", "text": part.text})
                    fallback += part.text
                elif isinstance(part, ImageContentPart):
                    images.append(_decode_image(part.image_url.url))
                    template_parts.append({"type": "image_url", "image_url": part.image_url.url})
                    fallback += getattr(self._processor, "boi_token", "<image>")
                else:
                    waveform, rate = _decode_audio(part)
                    audio.append(waveform)
                    sample_rates.append(rate)
                    template_parts.append(part.model_dump(mode="json"))
                    fallback += "<audio>"
            template_messages.append({"role": message.role, "content": template_parts})
            fallback_messages.append(f"{message.role}: {fallback}")
        chat_template = getattr(self._processor, "chat_template", None)
        if chat_template is not None:
            prompt = self._processor.apply_chat_template(
                template_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(fallback_messages) + "\nassistant:"
        parameters: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if images:
            parameters["images"] = images
        if audio:
            if len(set(sample_rates)) != 1:
                raise ConfigurationError("all audio inputs in one request must share a sample rate")
            parameters["audios"] = audio
            parameters["sampling_rate"] = sample_rates[0]
        inputs = self._processor(**parameters)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        text_config = getattr(self._model.config, "text_config", self._model.config)
        limit = getattr(text_config, "max_position_embeddings", prompt_tokens + request.max_tokens)
        if prompt_tokens + request.max_tokens > limit:
            raise ConfigurationError("multimodal prompt and max_tokens exceed the context limit")
        return inputs, prompt_tokens

    def generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        self._load()
        started = time.monotonic()
        with self._inference_lock:
            inputs, prompt_tokens = self._inputs(request)
            check_request_health(cancelled, started, timeout)
            if request.seed is not None:
                torch.manual_seed(request.seed)
            parameters: dict[str, Any] = {
                **inputs,
                "max_new_tokens": request.max_tokens,
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                parameters.update(
                    {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                    }
                )
            from transformers import StoppingCriteria, StoppingCriteriaList

            class RequestStop(StoppingCriteria):
                def __call__(
                    self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any
                ) -> bool:
                    del input_ids, scores, kwargs
                    try:
                        check_request_health(cancelled, started, timeout)
                    except Exception:
                        return True
                    return False

            parameters["stopping_criteria"] = StoppingCriteriaList([RequestStop()])
            with torch.inference_mode():
                output = self._model.generate(**parameters)
            check_request_health(cancelled, started, timeout)
            generated = output[0][prompt_tokens:]
            completion_ids = generated.tolist()
            text = cast(str, self._processor.decode(completion_ids, skip_special_tokens=True))
            eos = getattr(self._model.config, "eos_token_id", None)
            if eos is None:
                eos = getattr(
                    getattr(self._model.config, "text_config", None), "eos_token_id", None
                )
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            finish_reason: Literal["stop", "length"] = (
                "stop"
                if completion_ids
                and completion_ids[-1] in eos_ids
                and len(completion_ids) < request.max_tokens
                else "length"
            )
            yield GenerationEvent(delta=text)
            yield GenerationEvent(
                delta="",
                summary=GenerationSummary(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(completion_ids),
                    finish_reason=finish_reason,
                    duration_ms=(time.monotonic() - started) * 1000,
                ),
            )

    def close(self) -> None:
        self._closed = True
        if self._inference_lock.acquire(blocking=False):
            self._model = None
            self._processor = None
            self._inference_lock.release()
