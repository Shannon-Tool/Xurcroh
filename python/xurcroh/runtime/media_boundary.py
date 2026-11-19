from __future__ import annotations

import io
import secrets
from typing import Any, cast

import torch

from xurcroh.algebra.orthogonal import sample_orthogonal
from xurcroh.config import ModelDefinition
from xurcroh.errors import CapabilityError
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.protected_diffusion import ProtectedDiffusionPipeline
from xurcroh.runtime.vlm import _decode_image


class DiffusionMediaBoundary:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @classmethod
    def from_definition(cls, definition: ModelDefinition) -> DiffusionMediaBoundary:
        import diffusers

        pipeline_type = (
            diffusers.DiffusionPipeline
            if definition.pipeline_class is None
            else getattr(diffusers, definition.pipeline_class)
        )
        pipeline = pipeline_type.from_pretrained(
            definition.path,
            local_files_only=not definition.allow_download,
            trust_remote_code=False,
            revision=definition.revision,
        )
        return cls(pipeline)

    def _source_latent(
        self, request: ImageGenerationRequest, initial: torch.Tensor
    ) -> torch.Tensor:
        if request.image is None:
            return initial
        vae = getattr(self.pipeline, "vae", None)
        processor = getattr(self.pipeline, "image_processor", None)
        if vae is None or processor is None:
            raise CapabilityError("protected editing requires a client VAE and image processor")
        width, height = request.dimensions
        with torch.inference_mode():
            source_image = _decode_image(request.image).resize((width, height))
            pixels = processor.preprocess(source_image).to(dtype=initial.dtype)
            source = cast(torch.Tensor, vae.encode(pixels).latent_dist.mode())
        if source.shape != initial.shape:
            raise CapabilityError("source and generated latent shapes do not match")
        return source * (1 - request.strength) + initial * request.strength

    def raw_inputs(
        self, request: ImageGenerationRequest
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        seed = request.seed if request.seed is not None else secrets.randbits(63)
        unet = getattr(self.pipeline, "unet", None)
        if unet is None:
            raise CapabilityError("protected remote images require a UNet client boundary")
        channels = int(unet.config.in_channels)
        sample_size = unet.config.sample_size
        if not isinstance(sample_size, int):
            raise CapabilityError("protected remote images require a scalar UNet sample size")
        width, height = request.dimensions
        latent_width = max(1, width * sample_size // 8)
        latent_height = max(1, height * sample_size // 8)
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn(
            1, channels, latent_height, latent_width, generator=generator, dtype=torch.float32
        )
        encode_prompt = getattr(self.pipeline, "encode_prompt", None)
        if not callable(encode_prompt):
            raise CapabilityError("pipeline does not expose client-side prompt conditioning")
        conditioned = encode_prompt(
            request.prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = conditioned[0] if isinstance(conditioned, tuple) else conditioned
        if not isinstance(prompt_embeds, torch.Tensor):
            raise CapabilityError("prompt conditioner did not return a tensor")
        parameters = {
            "width": width,
            "height": height,
            "num_inference_steps": request.steps,
            "generator_seed": seed,
            "strength": request.strength,
            "output_type": "latent",
        }
        initial = self._source_latent(request, initial)
        return initial, prompt_embeds, parameters

    def raw_video_inputs(
        self, request: VideoGenerationRequest
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        image_request = ImageGenerationRequest(
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            steps=request.steps,
            seed=request.seed,
        )
        initial, prompt_embeds, parameters = self.raw_inputs(image_request)
        seed = request.seed if request.seed is not None else secrets.randbits(63)
        unet = getattr(self.pipeline, "unet", None)
        if unet is None:
            raise CapabilityError("protected remote video requires a UNet client boundary")
        channels = int(unet.config.in_channels)
        shape = (1, channels, request.frames, *initial.shape[-2:])
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn(shape, generator=generator, dtype=initial.dtype)
        parameters.update({"num_frames": request.frames, "fps": request.fps})
        return initial, prompt_embeds, parameters

    def prepare(
        self, request: ImageGenerationRequest
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        initial, prompt_embeds, parameters = self.raw_inputs(request)
        seed = int(parameters["generator_seed"])
        channels = initial.shape[1]
        basis = sample_orthogonal(channels, seed=seed + 10007, dtype=torch.float32)
        latent_width, latent_height = initial.shape[-2:]
        permutation = torch.randperm(
            latent_width * latent_height,
            generator=torch.Generator().manual_seed(seed + 10009),
        )
        protected = ProtectedDiffusionPipeline(self.pipeline, basis, permutation)
        return (
            {
                "latents": protected.protect(initial),
                "basis": basis,
                "permutation": permutation,
                "prompt_embeds": prompt_embeds,
            },
            parameters,
        )

    def decode(self, latent: torch.Tensor, request: ImageGenerationRequest) -> bytes:
        del request
        with torch.inference_mode():
            if hasattr(self.pipeline, "decode_latents"):
                pixels = self.pipeline.decode_latents(latent)
                images = self.pipeline.image_processor.postprocess(pixels, output_type="pil")
            else:
                vae = getattr(self.pipeline, "vae", None)
                processor = getattr(self.pipeline, "image_processor", None)
                if vae is None or processor is None:
                    raise CapabilityError("client boundary has no latent image decoder")
                scaling = getattr(vae.config, "scaling_factor", 1.0)
                pixels = vae.decode(latent / scaling).sample
                images = processor.postprocess(pixels, output_type="pil")
        if not images:
            raise CapabilityError("client image decoder returned no images")
        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    def prepare_video(
        self, request: VideoGenerationRequest
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        initial, prompt_embeds, parameters = self.raw_video_inputs(request)
        seed = int(parameters["generator_seed"])
        channels = initial.shape[1]
        basis = sample_orthogonal(channels, seed=seed + 10007, dtype=torch.float32)
        permutation = torch.randperm(
            initial.shape[2:].numel(),
            generator=torch.Generator().manual_seed(seed + 10009),
        )
        protected = ProtectedDiffusionPipeline(self.pipeline, basis, permutation)
        return (
            {
                "latents": protected.protect(initial),
                "basis": basis,
                "permutation": permutation,
                "prompt_embeds": prompt_embeds,
            },
            parameters,
        )

    def decode_video(self, latent: torch.Tensor, request: VideoGenerationRequest) -> bytes:
        with torch.inference_mode():
            if hasattr(self.pipeline, "decode_latents"):
                pixels = self.pipeline.decode_latents(latent)
            else:
                vae = getattr(self.pipeline, "vae", None)
                if vae is None:
                    raise CapabilityError("client boundary has no latent video decoder")
                scaling = getattr(vae.config, "scaling_factor", 1.0)
                pixels = vae.decode(latent / scaling).sample
            processor = getattr(self.pipeline, "image_processor", None)
            if processor is not None:
                pixels = processor.postprocess(pixels, output_type="np")
        import imageio.v2 as imageio
        import numpy as np

        frames = np.asarray(pixels)
        if frames.ndim == 5:
            if frames.shape[0] != 1:
                raise CapabilityError("client video decoder returned an unexpected batch size")
            frames = frames[0]
        if frames.ndim == 4 and frames.shape[-1] != 3 and frames.shape[0] == 3:
            frames = frames.transpose(1, 2, 3, 0)
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise CapabilityError("client video decoder did not return RGB frames")
        if frames.dtype != np.uint8:
            frames = (frames.clip(0, 1) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        with imageio.get_writer(buffer, format="mp4", fps=request.fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]
        return buffer.getvalue()
