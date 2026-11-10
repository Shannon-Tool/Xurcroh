from __future__ import annotations

import base64
import io
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xurcroh.algebra.orthogonal import sample_orthogonal
from xurcroh.config import ModelDefinition
from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.models.av import mux_video_audio
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.generation import check_request_health
from xurcroh.runtime.protected_diffusion import ProtectedDiffusionPipeline
from xurcroh.types import (
    ExecutionMetadata,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResponse,
    VideoGenerationResponse,
)


class LocalDiffusionEngine:
    def __init__(
        self,
        name: str,
        definition: ModelDefinition,
        worker_id: str,
        *,
        pipeline: Any = None,
    ) -> None:
        self.name = name
        self.definition = definition
        self.worker_id = worker_id
        self.engine_name = "diffusers"
        self.modality = definition.modality
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._closed = False
        self.protected_scheduler_steps = 0
        self.protected_model_calls = 0

    def _load(self) -> None:
        if self._closed:
            raise ModelLoadError(f"diffusion engine for {self.name} is closed")
        if self._pipeline is not None:
            return
        path = Path(self.definition.path)
        if not path.is_dir() and not self.definition.allow_download:
            raise ModelLoadError(f"diffusion model path is unavailable: {path}")
        try:
            from diffusers import DiffusionPipeline

            self._pipeline = DiffusionPipeline.from_pretrained(  # type: ignore[no-untyped-call]
                self.definition.path,
                local_files_only=not self.definition.allow_download,
                trust_remote_code=False,
            )
        except Exception as error:
            raise ModelLoadError(f"failed to load diffusion model {self.name}: {error}") from error

    def require_modality(self, expected: str) -> None:
        if self.modality != expected:
            raise CapabilityError(f"model {self.name} does not support {expected} generation")

    def _metadata(
        self, request_id: str, duration_ms: float, security_mode: str
    ) -> ExecutionMetadata:
        return ExecutionMetadata(
            request_id=request_id,
            engine=self.engine_name,
            worker_ids=[self.worker_id],
            duration_ms=max(duration_ms, 0.001),
            security_mode=security_mode,
        )

    def _protected_image(
        self,
        parameters: dict[str, Any],
        *,
        width: int,
        height: int,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> Any:
        unet = getattr(self._pipeline, "unet", None)
        if unet is None:
            raise CapabilityError("protected BCHW latents require a UNet pipeline")
        channels = int(unet.config.in_channels)
        sample_size = unet.config.sample_size
        if not isinstance(sample_size, int):
            raise CapabilityError("protected latents require a scalar UNet sample size")
        latent_width = max(1, width * sample_size // 8)
        latent_height = max(1, height * sample_size // 8)
        basis_seed = seed if seed is not None else secrets.randbits(63)
        basis = sample_orthogonal(channels, seed=basis_seed + 10007, dtype=torch.float32)
        permutation = torch.randperm(
            latent_width * latent_height,
            generator=torch.Generator().manual_seed(basis_seed + 10009),
        )
        protected = ProtectedDiffusionPipeline(self._pipeline, basis, permutation)
        initial = torch.randn(
            1,
            channels,
            latent_height,
            latent_width,
            generator=generator,
            dtype=torch.float32,
        )
        result = protected.generate(latents=protected.protect(initial), **parameters)
        self.protected_scheduler_steps += protected.scheduler_steps
        self.protected_model_calls += protected.model_calls
        initial.zero_()
        protected.close()
        return result

    def generate_protected_latent(
        self,
        parameters: dict[str, Any],
        *,
        latents: torch.Tensor,
        basis: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        self._load()
        protected = ProtectedDiffusionPipeline(self._pipeline, basis, permutation)
        values = dict(parameters)
        seed = values.pop("generator_seed", None)
        values["generator"] = torch.Generator().manual_seed(int(seed))
        values["output_type"] = "latent"
        with self._lock, torch.inference_mode():
            result = protected.generate(latents=latents, **values)
        latent = getattr(result, "latents", None)
        if not isinstance(latent, torch.Tensor):
            raise ModelLoadError("protected pipeline did not return a final latent")
        self.protected_scheduler_steps += protected.scheduler_steps
        self.protected_model_calls += protected.model_calls
        protected.close()
        return latent

    def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: threading.Event,
        timeout: float,
    ) -> ImageGenerationResponse:
        self.require_modality("image")
        self._load()
        started = time.monotonic()
        check_request_health(cancelled, started, timeout)
        width, height = request.dimensions
        generator = None if request.seed is None else torch.Generator().manual_seed(request.seed)
        parameters = {
            "prompt": request.prompt,
            "image": request.image,
            "width": width,
            "height": height,
            "num_inference_steps": request.steps,
            "generator": generator,
            "strength": request.strength,
        }
        with self._lock, torch.inference_mode():
            if self.definition.protect_latents:
                result = self._protected_image(
                    parameters,
                    width=width,
                    height=height,
                    seed=request.seed,
                    generator=generator,
                )
            else:
                result = self._pipeline(**parameters)
        check_request_health(cancelled, started, timeout)
        images = getattr(result, "images", None)
        if not images:
            raise ModelLoadError("diffusion pipeline returned no images")
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded.append(GeneratedImage(b64_json=base64.b64encode(buffer.getvalue()).decode()))
        return ImageGenerationResponse(
            created=int(time.time()),
            data=encoded,
            execution=self._metadata(
                request_id, (time.monotonic() - started) * 1000, security_mode
            ),
        )

    def generate_video(
        self,
        request: VideoGenerationRequest,
        *,
        request_id: str,
        security_mode: str,
        cancelled: threading.Event,
        timeout: float,
    ) -> VideoGenerationResponse:
        self.require_modality("video")
        self._load()
        started = time.monotonic()
        check_request_health(cancelled, started, timeout)
        width, height = (int(value) for value in request.size.split("x"))
        generator = None if request.seed is None else torch.Generator().manual_seed(request.seed)
        with self._lock, torch.inference_mode():
            result = self._pipeline(
                prompt=request.prompt,
                image=request.image,
                width=width,
                height=height,
                num_frames=request.frames,
                fps=request.fps,
                num_inference_steps=request.steps,
                generator=generator,
            )
        check_request_health(cancelled, started, timeout)
        frames = getattr(result, "frames", None)
        if frames is None:
            frames = getattr(result, "images", None)
        if frames is None or len(frames) == 0:
            raise ModelLoadError("video pipeline returned no frames")
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            frames = list(frames)
        elif len(frames) == 1:
            candidate = frames[0]
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach().cpu().numpy()
            if isinstance(candidate, np.ndarray) and candidate.ndim == 4:
                frames = list(candidate)
        if len(frames) != request.frames:
            raise ModelLoadError(
                f"video pipeline returned {len(frames)} frames instead of {request.frames}"
            )
        buffer = io.BytesIO()
        import imageio.v2 as imageio

        with imageio.get_writer(buffer, format="mp4", fps=request.fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))  # type: ignore[attr-defined]
        video_bytes = buffer.getvalue()
        audio = getattr(result, "audio", None)
        if audio is not None:
            video_bytes = mux_video_audio(video_bytes, bytes(audio))
        return VideoGenerationResponse(
            created=int(time.time()),
            data=[GeneratedVideo(b64_json=base64.b64encode(video_bytes).decode())],
            execution=self._metadata(
                request_id, (time.monotonic() - started) * 1000, security_mode
            ),
        )

    def close(self) -> None:
        self._closed = True
        self._pipeline = None
