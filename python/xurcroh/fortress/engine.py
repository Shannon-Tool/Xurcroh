from __future__ import annotations

import inspect
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import (
    CapabilityError,
    ConfigurationError,
    ModelLoadError,
)
from xurcroh.fortress.combiner import FortressCombiner
from xurcroh.fortress.model_compiler import FortressCheckpoint, compile_fortress_checkpoint
from xurcroh.models.vlm import GemmaAudioEmbedder, GemmaVisionEmbedder
from xurcroh.requests import AudioContentPart, ImageContentPart, TextContentPart

if TYPE_CHECKING:
    from xurcroh.fortress.compiler import CompiledOperation
    from xurcroh.fortress.transport import (
        FortressGridRuntime,
        FortressRoleClient,
        MutualTLSIdentity,
        RoleEndpoint,
    )
    from xurcroh.runtime.generation import GenerationEvent
from xurcroh.prompting import render_chat_prompt
from xurcroh.requests import CompletionRequest


class FortressEngine:
    def __init__(
        self,
        name: str,
        definition: ModelDefinition,
        workers: list[WorkerConfig],
        recover_role: Callable[[str, int], int] | None = None,
    ) -> None:
        self.name = name
        self.definition = definition
        self.workers = workers
        self._recover_role = recover_role
        self.worker_id = workers[0].name
        self.worker_ids = [worker.name for worker in workers]
        self.engine_name = (
            "fortress-local" if workers[0].endpoint == "local://cpu" else "fortress-grid"
        )
        self._tokenizer: Any = None
        self._tokenizer_tempdir: Any = None
        self._processor: Any = None
        self._stop_token_ids: set[int] = set()
        self._compiled: FortressCheckpoint | None = None
        self._combiner: FortressCombiner | None = None
        self._clients: dict[str, FortressRoleClient] = {}
        self._grid_runtime: FortressGridRuntime | None = None
        self._grid_runtimes: list[FortressGridRuntime] = []
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._closed = False

    def require_modality(self, expected: str) -> None:
        supported = {"text", "vlm"} if self.definition.modality == "vlm" else {"text"}
        if expected not in supported:
            raise CapabilityError(f"Fortress engine does not support {expected}")

    @contextmanager
    def _inference_scope(
        self, started: float, timeout: float, cancelled: threading.Event
    ) -> Iterator[None]:
        from xurcroh.fortress.transport import execution_health
        from xurcroh.runtime.generation import bounded_execution

        with (
            bounded_execution(self._inference_lock, cancelled, started, timeout, self.name),
            execution_health(cancelled, started + timeout),
        ):
            yield

    def _identity(self) -> MutualTLSIdentity:
        from xurcroh.fortress.transport import MutualTLSIdentity

        labels = self.workers[0].labels
        try:
            client_certificate = Path(labels["client_certificate_path"]).read_bytes()
            client_key = Path(labels["client_key_path"]).read_bytes()
            ca = Path(labels["ca_path"]).read_bytes()
        except (KeyError, OSError) as error:
            raise ConfigurationError(
                "Fortress workers require client_certificate_path, client_key_path, and ca_path"
            ) from error
        return MutualTLSIdentity(
            certificates=(client_certificate, ca),
            private_key=client_key,
            pinned_ca=ca,
            server_name=labels.get("server_name", "localhost"),
        )

    def _roles(self, operation_name: str | None = None) -> list[RoleEndpoint]:
        from xurcroh.fortress.transport import RoleEndpoint

        roles = []
        for worker in self.workers:
            assigned = {name.strip() for name in worker.labels.get("operations", "*").split(",")}
            if (
                operation_name is not None
                and operation_name not in assigned
                and "*" not in assigned
            ):
                continue
            try:
                if operation_name is not None and "operation_roles" in worker.labels:
                    assigned_role = json.loads(worker.labels["operation_roles"])[operation_name]
                    if not isinstance(assigned_role, list) or len(assigned_role) != 2:
                        raise ValueError("operation role must contain two indexes")
                    if any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in assigned_role
                    ):
                        raise ValueError("operation role indexes must be integers")
                    activation_index = assigned_role[0]
                    weight_index = assigned_role[1]
                else:
                    activation_index = int(worker.labels["activation_index"])
                    weight_index = int(worker.labels["weight_index"])
                recipient_public = bytes.fromhex(worker.labels["recipient_public"])
                address = (worker.endpoint or "").removeprefix("quic://")
            except (KeyError, TypeError, ValueError) as error:
                raise ConfigurationError(
                    f"worker {worker.name} lacks Fortress role indexes or recipient public key"
                ) from error
            if not address:
                raise ConfigurationError(f"worker {worker.name} requires a QUIC endpoint")
            roles.append(
                RoleEndpoint(
                    activation_index=activation_index,
                    weight_index=weight_index,
                    address=address,
                    recipient_public=recipient_public,
                    provider_id=worker.name,
                    server_name=worker.labels.get("server_name"),
                    generation=int(worker.labels.get("restart_generation", "0")),
                )
            )
        if operation_name is not None and not roles:
            raise ConfigurationError(f"operation {operation_name!r} has no assigned Fortress roles")
        return roles

    def _grid_clients(
        self, operations: Mapping[str, CompiledOperation], *, namespace: str | None = None
    ) -> dict[str, FortressRoleClient]:
        from xurcroh.fortress.transport import FortressGridRuntime

        identity = self._identity()
        frame_bytes = int(self.workers[0].labels.get("frame_bytes", 16_384))
        cover_interval = float(self.workers[0].labels.get("cover_interval", 1.0))
        seeded_activations = self.workers[0].labels.get(
            "seeded_activations", "true"
        ).lower() not in {"0", "false", "no"}
        runtimes: dict[tuple[str, ...], FortressGridRuntime] = {}
        clients: dict[str, FortressRoleClient] = {}
        for name, operation in operations.items():
            qualified = name if namespace is None else f"{namespace}:{name}"
            roles = self._roles(qualified)
            group = tuple(role.provider_id for role in roles)
            runtime = runtimes.get(group)
            if runtime is None:
                runtime = FortressGridRuntime(
                    roles,
                    identity,
                    frame_bytes=frame_bytes,
                    cover_interval=cover_interval,
                    seeded_activations=seeded_activations,
                    recover_role=self._recover_role,
                )
                runtimes[group] = runtime
                self._grid_runtimes.append(runtime)
                if self._grid_runtime is None:
                    self._grid_runtime = runtime
            clients[name] = runtime.client(operation, qualified)
        return clients

    def _load(self) -> None:
        if self._closed:
            raise ModelLoadError(f"model {self.name!r} Fortress engine is closed")
        if self._combiner is not None:
            return
        with self._load_lock:
            if self._combiner is not None:
                return
            try:
                from transformers import AutoTokenizer, GenerationConfig

                options = {
                    "local_files_only": not self.definition.allow_download,
                    "trust_remote_code": False,
                    "revision": self.definition.revision,
                }
                tokenizer_path = self.definition.path
                if self.definition.source_format == "gguf":
                    import tempfile

                    from xurcroh.formats import open_checkpoint
                    from xurcroh.formats.gguf_tokenizer import materialize_client_tokenizer

                    self._tokenizer_tempdir = tempfile.TemporaryDirectory(
                        prefix="xurcroh-gguf-tokenizer-"
                    )
                    with open_checkpoint(self.definition.path) as source:
                        tokenizer_path = materialize_client_tokenizer(
                            source, self._tokenizer_tempdir.name
                        ).as_posix()
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **options)
                try:
                    generation = GenerationConfig.from_pretrained(
                        tokenizer_path,
                        local_files_only=not self.definition.allow_download,
                        trust_remote_code=False,
                        revision=self.definition.revision or "main",
                    )
                    eos_ids = generation.eos_token_id
                except OSError:
                    eos_ids = None
                if eos_ids is None:
                    eos_ids = self._tokenizer.eos_token_id
                self._stop_token_ids = {
                    int(value)
                    for value in (eos_ids if isinstance(eos_ids, list) else [eos_ids])
                    if value is not None
                }
                if self.definition.source_format == "gguf":
                    from xurcroh.formats import open_checkpoint

                    with open_checkpoint(self.definition.path) as source:
                        checkpoint_config = source.config
                else:
                    checkpoint_config = json.loads(
                        (Path(self.definition.path) / "config.json").read_text()
                    )
                official_vlm = self.definition.modality == "vlm" and checkpoint_config.get(
                    "model_type"
                ) in {"gemma4", "qwen3_5"}
                if official_vlm:
                    from transformers import AutoProcessor

                    self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                        self.definition.path, tokenizer=self._tokenizer, **options
                    )
                from xurcroh.fortress.model_compiler import load_reap_saliency

                self._compiled = compile_fortress_checkpoint(
                    self.definition.path,
                    seed=self.definition.fortress_seed,
                    share_count=self.definition.fortress_shares,
                    materialize_shares=self.engine_name == "fortress-local",
                    retain_quantized=self.engine_name == "fortress-local",
                    model_dtype=(
                        torch.float32 if self.engine_name == "fortress-grid" else torch.float64
                    ),
                    load_full_multimodal=official_vlm,
                    reap_keep_fraction=self.definition.reap_keep_fraction,
                    reap_saliency=(
                        load_reap_saliency(self.definition.reap_saliency_path)
                        if self.definition.reap_saliency_path is not None
                        else None
                    ),
                )
                compiled = self._compiled
                clients: dict[str, FortressRoleClient] = {}
                if self.engine_name == "fortress-grid":
                    clients = self._grid_clients(compiled.operations)
                self._clients = clients
                if compiled.model is not None and clients:
                    from xurcroh.fortress.module import connect_grid_clients

                    model_clients = {
                        **clients,
                        **{
                            name.replace("model.language_model.", "model.", 1): client
                            for name, client in clients.items()
                        },
                    }
                    connect_grid_clients(compiled.model, model_clients)
                self._combiner = FortressCombiner(
                    layers=list(compiled.layers),
                    operations=compiled.operations,
                    query_heads=compiled.query_heads,
                    kv_heads=compiled.kv_heads,
                    final_norm=compiled.final_norm,
                    lm_head=compiled.lm_head,
                    rope_theta=compiled.rope_theta,
                    activation_bits=compiled.activation_bits,
                    grid_clients=clients or None,
                )
            except Exception:
                self.close()
                raise

    def _text_hidden(self, text: str) -> torch.Tensor:
        compiled = self._compiled
        assert compiled is not None
        input_ids = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        if input_ids.numel() == 0:
            return compiled.embedding.new_empty((0, compiled.embedding.shape[1]))
        if int(input_ids.max()) >= compiled.embedding.shape[0]:
            raise ModelLoadError("tokenizer emits IDs outside the model vocabulary")
        return compiled.embedding[input_ids[0]]

    def _prepare_hidden(self, request: CompletionRequest) -> tuple[torch.Tensor, int]:
        from xurcroh.runtime.vlm import _decode_audio, _decode_image

        self._load()
        compiled = self._compiled
        assert compiled is not None
        if all(isinstance(message.content, str) for message in request.messages):
            prompt = render_chat_prompt(self._tokenizer, request.messages)
            hidden = self._text_hidden(prompt)
            if hidden.shape[0] == 0:
                raise ConfigurationError("Fortress tokenizer produced an empty prompt")
            return hidden, hidden.shape[0]
        chunks = []
        for message in request.messages:
            parts = (
                [TextContentPart(type="text", text=message.content)]
                if isinstance(message.content, str)
                else message.content
            )
            for part in parts:
                if isinstance(part, TextContentPart):
                    chunks.append(self._text_hidden(f"{message.role}: {part.text}"))
                elif isinstance(part, ImageContentPart):
                    projection = compiled.media_projections.get("vision")
                    if projection is None:
                        raise CapabilityError("checkpoint has no client vision projection")
                    image = _decode_image(part.image_url.url)
                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255
                    chunks.append(GemmaVisionEmbedder(projection).embed(pixels))
                elif isinstance(part, AudioContentPart):
                    projection = compiled.media_projections.get("audio")
                    if projection is None:
                        raise CapabilityError("checkpoint has no client audio projection")
                    waveform, _ = _decode_audio(part)
                    chunks.append(GemmaAudioEmbedder(projection).embed(torch.from_numpy(waveform)))
                else:
                    raise ConfigurationError("unsupported multimodal content part")
        if not chunks:
            raise ConfigurationError("multimodal request contains no embeddings")
        hidden = torch.cat(chunks, dim=0)
        return hidden, hidden.shape[0]

    def _generate_official_model(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        from xurcroh.runtime.generation import (
            GenerationEvent,
            GenerationSummary,
            check_request_health,
            sample_next_token,
            stopping_strings,
        )

        compiled = self._compiled
        assert compiled is not None and compiled.model is not None
        started = time.monotonic()
        with self._inference_scope(started, timeout, cancelled):
            prompt = render_chat_prompt(self._tokenizer, request.messages)
            input_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ]
            if input_ids.numel() == 0:
                raise ConfigurationError("Fortress tokenizer produced an empty prompt")
            if int(input_ids.max()) >= compiled.embedding.shape[0]:
                raise ModelLoadError("tokenizer emits IDs outside the model vocabulary")
            current = input_ids
            past = None
            shared = None
            completion_ids: list[int] = []
            text = ""
            finish_reason: Literal["stop", "length"] = "length"
            stops = stopping_strings(request)
            for _ in range(request.max_tokens):
                check_request_health(cancelled, started, timeout)
                parameters = {
                    "input_ids": current,
                    "past_key_values": past,
                    "return_shared_kv_states": True,
                    "use_cache": True,
                }
                if shared is not None:
                    parameters["shared_kv_states"] = shared
                with torch.inference_mode():
                    outputs = compiled.model(**parameters)
                next_token = sample_next_token(outputs.logits[:, -1, :], request)
                token_id = int(next_token.item())
                completion_ids.append(token_id)
                updated_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
                delta = updated_text[len(text) :]
                text = updated_text
                past = outputs.past_key_values
                shared = getattr(outputs, "shared_kv_states", None)
                current = next_token
                yield GenerationEvent(delta=delta)
                if token_id in self._stop_token_ids:
                    finish_reason = "stop"
                    break
                if stops and any(text.endswith(stop) for stop in stops):
                    finish_reason = "stop"
                    break
            check_request_health(cancelled, started, timeout)
            yield GenerationEvent(
                delta="",
                summary=GenerationSummary(
                    text=text,
                    prompt_tokens=input_ids.shape[-1],
                    completion_tokens=len(completion_ids),
                    finish_reason=finish_reason,
                    duration_ms=(time.monotonic() - started) * 1000,
                ),
            )

    def _generate_official_vlm(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        from xurcroh.runtime.generation import (
            GenerationEvent,
            GenerationSummary,
            check_request_health,
            stopping_strings,
        )
        from xurcroh.runtime.vlm import _decode_audio, _decode_image

        compiled = self._compiled
        assert compiled is not None and compiled.model is not None and self._processor is not None
        started = time.monotonic()
        with self._inference_scope(started, timeout, cancelled):
            images = []
            audio = []
            sample_rates = []
            template_messages = []
            for message in request.messages:
                parts = (
                    [TextContentPart(type="text", text=message.content)]
                    if isinstance(message.content, str)
                    else message.content
                )
                template_parts = []
                for part in parts:
                    if isinstance(part, TextContentPart):
                        template_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContentPart):
                        images.append(_decode_image(part.image_url.url))
                        template_parts.append(
                            {"type": "image_url", "image_url": part.image_url.url}
                        )
                    elif isinstance(part, AudioContentPart):
                        waveform, rate = _decode_audio(part)
                        audio.append(waveform)
                        sample_rates.append(rate)
                        template_parts.append(part.model_dump(mode="json"))
                template_messages.append({"role": message.role, "content": template_parts})
            if self._processor.chat_template is None:
                raise CapabilityError("official VLM processor has no chat template")
            prompt = self._processor.apply_chat_template(
                template_messages, tokenize=False, add_generation_prompt=True
            )
            parameters: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
            if images:
                parameters["images"] = images
            if audio:
                if len(set(sample_rates)) != 1:
                    raise ConfigurationError("audio inputs must share one sample rate")
                audio_parameter = (
                    "audio"
                    if "audio" in inspect.signature(self._processor.__call__).parameters
                    else "audios"
                )
                parameters[audio_parameter] = audio
                parameters["sampling_rate"] = sample_rates[0]
            inputs = self._processor(**parameters)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            check_request_health(cancelled, started, timeout)
            generation: dict[str, Any] = {
                **inputs,
                "max_new_tokens": request.max_tokens,
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                generation.update(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
            if request.seed is not None:
                torch.manual_seed(request.seed)
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

            generation["stopping_criteria"] = StoppingCriteriaList([RequestStop()])
            with torch.inference_mode():
                output = compiled.model.generate(**generation)
            check_request_health(cancelled, started, timeout)
            completion_ids = output[0][prompt_tokens:].tolist()
            text = self._processor.decode(completion_ids, skip_special_tokens=True)
            stops = stopping_strings(request)
            stop_positions = [text.find(stop) for stop in stops if stop in text]
            stopped_on_text = bool(stop_positions)
            if stopped_on_text:
                text = text[: min(stop_positions)]
            finish_reason: Literal["stop", "length"] = (
                "stop"
                if stopped_on_text
                or (
                    completion_ids
                    and completion_ids[-1] in self._stop_token_ids
                    and len(completion_ids) < request.max_tokens
                )
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

    def generate(
        self,
        request: CompletionRequest,
        *,
        cancelled: threading.Event,
        timeout: float,
    ) -> Iterator[GenerationEvent]:
        from xurcroh.runtime.generation import (
            GenerationEvent,
            GenerationSummary,
            check_request_health,
            sample_next_token,
            stopping_strings,
        )

        self._load()
        assert self._combiner is not None
        compiled = self._compiled
        assert compiled is not None
        if compiled.model is not None and all(
            isinstance(message.content, str) for message in request.messages
        ):
            yield from self._generate_official_model(request, cancelled=cancelled, timeout=timeout)
            return
        if compiled.model is not None and self._processor is not None:
            yield from self._generate_official_vlm(request, cancelled=cancelled, timeout=timeout)
            return
        started = time.monotonic()
        with self._inference_scope(started, timeout, cancelled):
            hidden, prompt_tokens = self._prepare_hidden(request)
            state_key = f"{self.name}:{uuid.uuid4().hex}"
            completion_ids: list[int] = []
            text = ""
            finish_reason: Literal["stop", "length"] = "length"
            stops = stopping_strings(request)
            try:
                hidden = self._combiner.run_cached(state_key, hidden)
                for _ in range(request.max_tokens):
                    check_request_health(cancelled, started, timeout)
                    next_token = sample_next_token(self._combiner.logits(hidden[-1:]), request)
                    token_id = int(next_token.item())
                    completion_ids.append(token_id)
                    updated_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
                    delta = updated_text[len(text) :]
                    text = updated_text
                    yield GenerationEvent(delta=delta)
                    if token_id in self._stop_token_ids:
                        finish_reason = "stop"
                        break
                    if stops and any(text.endswith(stop) for stop in stops):
                        finish_reason = "stop"
                        break
                    hidden = self._combiner.run_cached(state_key, compiled.embedding[next_token[0]])
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
            finally:
                self._combiner.drop_request(state_key)

    def transport_metrics(self) -> dict[str, int]:
        clients = list(self._clients.values())
        return {
            "fortress_role_requests": sum(client.request_count for client in clients),
            "fortress_bytes_sent": sum(client.bytes_sent for client in clients),
            "fortress_bytes_received": sum(client.bytes_received for client in clients),
            "fortress_cover_requests": sum(client.cover_request_count for client in clients),
            "fortress_cover_bytes_sent": sum(client.cover_bytes_sent for client in clients),
            "fortress_cover_bytes_received": sum(client.cover_bytes_received for client in clients),
            "fortress_reconnections": sum(
                transport.reconnect_count
                for client in clients
                for transport in client._transports.values()
            ),
            "fortress_cover_failures": sum(client.cover_failure_count for client in clients),
        }

    def close(self) -> None:
        self._closed = True
        try:
            for client in self._clients.values():
                client.close()
        finally:
            self._clients.clear()
            for runtime in self._grid_runtimes:
                runtime.close()
            self._grid_runtimes.clear()
            self._grid_runtime = None
        self._combiner = None
        self._compiled = None
        self._tokenizer = None
        if self._tokenizer_tempdir is not None:
            self._tokenizer_tempdir.cleanup()
            self._tokenizer_tempdir = None
        self._processor = None
        self._stop_token_ids.clear()
