from __future__ import annotations

import base64
import inspect
import io
import time
from typing import TYPE_CHECKING, Any

import torch

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.fortress.diffusers_loader import compile_diffusers_component
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


class FortressDiffusionEngine(FortressEngine):
    def __init__(self, name: str, definition: ModelDefinition, workers: list[WorkerConfig]) -> None:
        super().__init__(name, definition, workers)
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
                from xurcroh.runtime.media_boundary import DiffusionMediaBoundary

                boundary = DiffusionMediaBoundary.from_definition(self.definition)
                component = "transformer" if hasattr(boundary.pipeline, "transformer") else "unet"
                module = getattr(boundary.pipeline, component)
                operations = compile_diffusers_component(
                    self.definition.path,
                    component=component,
                    seed=self.definition.fortress_seed,
                    share_count=self.definition.fortress_shares,
                )
                clients: dict[str, FortressRoleClient] = {}
                if self.engine_name == "fortress-diffusion-grid":
                    from xurcroh.fortress.transport import FortressGridRuntime

                    identity = self._identity()
                    roles = self._roles()
                    labels = self.workers[0].labels
                    self._grid_runtime = FortressGridRuntime(
                        roles,
                        identity,
                        frame_bytes=int(labels.get("frame_bytes", 16_384)),
                        cover_interval=float(labels.get("cover_interval", 1.0)),
                        seeded_activations=labels.get("seeded_activations", "true").lower()
                        not in {"0", "false", "no"},
                    )
                    clients = {
                        name: self._grid_runtime.client(operation, name)
                        for name, operation in operations.items()
                    }
                self._clients = clients
                replace_linear_layers(
                    module,
                    operations,
                    grid_clients=clients or None,
                )
                self._boundary = boundary
            except Exception:
                self.close()
                raise

    def _metadata(self, request_id: str, started: float, security_mode: str) -> ExecutionMetadata:
        return ExecutionMetadata(
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
        from xurcroh.runtime.generation import check_request_health
        from xurcroh.runtime.vlm import _decode_image

        self._load()
        assert self._boundary is not None
        started = time.monotonic()
        seed = self._seed(request)
        supports_latent_decode = hasattr(self._boundary.pipeline, "decode_latents") or hasattr(
            self._boundary.pipeline, "vae"
        )
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

        def health(_pipeline: Any, _index: int, _timestep: Any, kwargs: Any) -> Any:
            check_request_health(cancelled, started, timeout)
            return kwargs

        parameters["callback_on_step_end"] = health
        signature = inspect.signature(self._boundary.pipeline.__call__).parameters
        parameters = {key: value for key, value in parameters.items() if key in signature}
        result = self._boundary.pipeline(**parameters)
        if supports_latent_decode:
            encoded = self._boundary.decode(result.images, request)
        else:
            encoded = encode_png(result.images[0])
        check_request_health(cancelled, started, timeout)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[GeneratedImage(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode),
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
        from xurcroh.runtime.generation import check_request_health

        self._load()
        assert self._boundary is not None
        started = time.monotonic()
        seed = self._seed(request)
        if request.image is not None:
            raise CapabilityError(
                "Fortress image-to-video requires a video-specific source adapter"
            )

        def health(_pipeline: Any, _index: int, _timestep: Any, kwargs: Any) -> Any:
            check_request_health(cancelled, started, timeout)
            return kwargs

        width, height = (int(value) for value in request.size.split("x"))
        parameters = {
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "num_frames": request.frames,
            "fps": request.fps,
            "num_inference_steps": request.steps,
            "generator": torch.Generator().manual_seed(seed),
            "output_type": "latent",
            "callback_on_step_end": health,
        }
        signature = inspect.signature(self._boundary.pipeline.__call__).parameters
        parameters = {key: value for key, value in parameters.items() if key in signature}
        result = self._boundary.pipeline(**parameters)
        frames = getattr(result, "frames", None)
        if frames is None:
            raise CapabilityError("video pipeline did not return latent frames")
        encoded = self._boundary.decode_video(frames, request)
        check_request_health(cancelled, started, timeout)
        return VideoGenerationResponse(
            created=int(time.time()),
            data=[GeneratedVideo(b64_json=base64.b64encode(encoded).decode())],
            execution=self._metadata(request_id, started, security_mode),
        )


def encode_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
