from __future__ import annotations

import base64
import inspect
import io
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.fortress.engine import FortressEngine
from xurcroh.fortress.module import replace_linear_layers
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest

if TYPE_CHECKING:
    from xurcroh.fortress.transport import FortressRoleClient
    from xurcroh.runtime.media_boundary import DiffusionMediaBoundary
from xurcroh.types import (
    ExecutionMetadata,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResponse,
    VideoGenerationResponse,
)


def _accepts_parameter(parameters: Mapping[str, inspect.Parameter], name: str) -> bool:
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


class FortressDiffusionEngine(FortressEngine):
    def __init__(
        self,
        name: str,
        definition: ModelDefinition,
        workers: list[WorkerConfig],
        recover_role: Callable[[str, int], int] | None = None,
    ) -> None:
        super().__init__(name, definition, workers, recover_role)
        self.engine_name = (
            "fortress-diffusion-local"
            if workers[0].endpoint == "local://cpu"
            else "fortress-diffusion-grid"
        )
        self._boundary: DiffusionMediaBoundary | None = None

    def require_modality(self, expected: str) -> None:
        if expected not in {"image", "video"}:
            raise CapabilityError("Fortress diffusion engine supports image and video generation")

    def _load(self) -> None:
        if self._closed:
            raise ModelLoadError(f"model {self.name!r} Fortress diffusion engine is closed")
        if self._boundary is not None:
            return
        with self._load_lock:
            if self._boundary is not None:
                return
            try:
                from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
                from xurcroh.fortress.media_components import compile_media_checkpoint
                from xurcroh.runtime.media_boundary import DiffusionMediaBoundary

                boundary = DiffusionMediaBoundary.from_definition(self.definition)
                adapter = MediaArchitectureAdapter.from_pipeline(
                    boundary.pipeline, self.definition.modality
                )
                compiled = compile_media_checkpoint(self.definition, adapter)
                clients: dict[str, FortressRoleClient] = {}
                for component, operations in compiled.items():
                    component_clients: dict[str, FortressRoleClient] = {}
                    if self.engine_name == "fortress-diffusion-grid":
                        component_clients = self._grid_clients(operations, namespace=component)
                    for name, client in component_clients.items():
                        clients[f"{component}:{name}"] = client
                    replace_linear_layers(
                        getattr(boundary.pipeline, component),
                        operations,
                        grid_clients=component_clients or None,
                    )
                self._clients = clients
                self._boundary = boundary
            except Exception:
                self.close()
                raise

    def _metadata(
        self, request_id: str, started: float, security_mode: str, seed: int
    ) -> ExecutionMetadata:
        return ExecutionMetadata(
            seed=seed,
            request_id=request_id,
            engine=self.engine_name,
            worker_ids=self.worker_ids,
            duration_ms=max((time.monotonic() - started) * 1000, 0.001),
            security_mode=security_mode,
        )

    def _seed(self, request: ImageGenerationRequest | VideoGenerationRequest) -> int:
        import secrets

        return request.seed if request.seed is not None else secrets.randbits(63)

    def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: Any,
        timeout: float,
    ) -> ImageGenerationResponse:
        from xurcroh.fortress.transport import execution_health
        from xurcroh.runtime.generation import check_request_health
        from xurcroh.runtime.media_boundary import needs_pipeline_decoder
        from xurcroh.runtime.vlm import _decode_image

        self._load()
        assert self._boundary is not None
        started = time.monotonic()
        seed = self._seed(request)
        supports_latent_decode = (
            hasattr(self._boundary.pipeline, "decode_latents")
            or hasattr(self._boundary.pipeline, "vae")
        ) and not needs_pipeline_decoder(self._boundary.pipeline)
        parameters: dict[str, Any] = {
            "prompt": request.prompt,
            "width": request.dimensions[0],
            "height": request.dimensions[1],
            "num_inference_steps": request.steps,
            "generator": torch.Generator().manual_seed(seed),
            "output_type": "latent" if supports_latent_decode else "pil",
        }
        if request.image is not None:
            parameters["image"] = _decode_image(request.image)
            parameters["strength"] = request.strength
        if request.mask is not None:
            signature = inspect.signature(self._boundary.pipeline.__call__).parameters
            mask_parameter = next(
                (name for name in ("mask_image", "mask") if _accepts_parameter(signature, name)),
                None,
            )
            if mask_parameter is None:
                raise CapabilityError("image editing pipeline does not accept a mask")
            parameters[mask_parameter] = _decode_image(request.mask)

        def health(_pipeline: Any, _index: int, _timestep: Any, kwargs: Any) -> Any:
            check_request_health(cancelled, started, timeout)
            return kwargs

        parameters["callback_on_step_end"] = health
        signature = inspect.signature(self._boundary.pipeline.__call__).parameters
        if request.image is not None and not _accepts_parameter(signature, "image"):
            raise CapabilityError("image pipeline does not support image conditioning")
        parameters = {
            key: value for key, value in parameters.items() if _accepts_parameter(signature, key)
        }
        with execution_health(cancelled, started + timeout):
            result = self._boundary.pipeline(**parameters)
        if supports_latent_decode:
            encoded = self._boundary.decode(result.images, request)
        else:
            encoded = encode_png(result.images[0])
        check_request_health(cancelled, started, timeout)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[GeneratedImage(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode, seed),
        )

    def generate_video(
        self,
        request: VideoGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: Any,
        timeout: float,
    ) -> VideoGenerationResponse:
        from xurcroh.fortress.transport import execution_health
        from xurcroh.runtime.generation import check_request_health
        from xurcroh.runtime.media_boundary import encode_video_frames, needs_pipeline_decoder
        from xurcroh.runtime.vlm import _decode_image

        self._load()
        assert self._boundary is not None
        started = time.monotonic()
        seed = self._seed(request)

        def health(_pipeline: Any, _index: int, _timestep: Any, kwargs: Any) -> Any:
            check_request_health(cancelled, started, timeout)
            return kwargs

        width, height = (int(value) for value in request.size.split("x"))
        native_decode = needs_pipeline_decoder(self._boundary.pipeline)
        parameters = {
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "num_frames": request.frames,
            "fps": request.fps,
            "num_inference_steps": request.steps,
            "generator": torch.Generator().manual_seed(seed),
            "output_type": "np" if native_decode else "latent",
            "callback_on_step_end": health,
        }
        if request.image is not None:
            parameters["image"] = _decode_image(request.image)
        signature = inspect.signature(self._boundary.pipeline.__call__).parameters
        if request.image is not None and not _accepts_parameter(signature, "image"):
            raise CapabilityError("video pipeline does not support image-to-video conditioning")
        parameters = {
            key: value for key, value in parameters.items() if _accepts_parameter(signature, key)
        }
        with execution_health(cancelled, started + timeout):
            result = self._boundary.pipeline(**parameters)
        frames = getattr(result, "frames", None)
        if frames is None:
            raise CapabilityError("video pipeline did not return frames")
        encoded = (
            encode_video_frames(frames, request.fps)
            if native_decode
            else self._boundary.decode_video(frames, request)
        )
        check_request_health(cancelled, started, timeout)
        return VideoGenerationResponse(
            created=int(time.time()),
            data=[GeneratedVideo(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode, seed),
        )


def encode_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
