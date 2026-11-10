from __future__ import annotations

import base64
import io
import threading

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, VaeImageProcessor
from PIL import Image

from xurcroh.config import ModelDefinition
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.diffusion import LocalDiffusionEngine


class TinyNeuralPipeline:
    def __init__(self) -> None:
        self.unet = UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 16),
            norm_num_groups=1,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=4)
        self.unet.eval()
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            sample_size=8,
            block_out_channels=(8,),
            norm_num_groups=1,
            scaling_factor=1.0,
            force_upcast=False,
        )
        self.vae.eval()
        self.image_processor = VaeImageProcessor(vae_scale_factor=1)

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        del device, num_images_per_prompt, do_classifier_free_guidance
        return (torch.tensor([[[float(sum(prompt.encode()))]]]).repeat(1, 1, 1280),)

    def decode_latents(self, latents):
        return self.vae.decode(latents).sample

    def __call__(
        self,
        *,
        prompt=None,
        image=None,
        width,
        height,
        num_inference_steps,
        generator,
        num_frames=1,
        latents=None,
        prompt_embeds=None,
        output_type="pil",
        **kwargs,
    ):
        del image, width, height, kwargs
        conditioning = (
            prompt_embeds
            if prompt_embeds is not None
            else torch.tensor([[[float(sum(prompt.encode()))]]]).repeat(1, 1, 1280)
        )
        self.received_latent = latents
        latent = torch.randn(1, 4, 8, 8, generator=generator) if latents is None else latents
        for timestep in self.scheduler.timesteps[:num_inference_steps]:
            noise = self.unet(latent, timestep, encoder_hidden_states=conditioning).sample
            latent = self.scheduler.step(noise, timestep, latent).prev_sample
        self.final_latent = latent
        if output_type == "latent":
            return type("Output", (), {"latents": latent})()
        with torch.inference_mode():
            decoded = self.vae.decode(latent).sample
            pixels = self.image_processor.postprocess(decoded, output_type="np")[0]
        pixels = (pixels * 255).round().astype(np.uint8)
        return type(
            "Output", (), {"images": [Image.fromarray(pixels) for _ in range(num_frames)]}
        )()


def test_image_engine_returns_real_model_computed_png() -> None:
    engine = LocalDiffusionEngine(
        "tiny",
        ModelDefinition(path="tiny", modality="image"),
        "local",
        pipeline=TinyNeuralPipeline(),
    )
    response = engine.generate_image(
        ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=2, seed=1),
        request_id="image-test",
        security_mode="strict",
        cancelled=threading.Event(),
        timeout=60,
    )
    data = base64.b64decode(response.data[0].b64_json)
    image = Image.open(io.BytesIO(data))
    assert image.size == (8, 8)
    assert np.asarray(image).std() > 0


def test_video_engine_returns_real_model_computed_mp4() -> None:
    engine = LocalDiffusionEngine(
        "tiny",
        ModelDefinition(path="tiny", modality="video"),
        "local",
        pipeline=TinyNeuralPipeline(),
    )
    response = engine.generate_video(
        VideoGenerationRequest(model="tiny", prompt="waves", size="8x8", steps=2, seed=2),
        request_id="video-test",
        security_mode="strict",
        cancelled=threading.Event(),
        timeout=60,
    )
    data = base64.b64decode(response.data[0].b64_json)
    assert data.startswith(b"\x00\x00\x00")
    assert b"ftyp" in data[:32]


def test_video_engine_rejects_missing_frames() -> None:
    import pytest

    from xurcroh.errors import ModelLoadError

    class SingleFramePipeline:
        def __call__(self, **kwargs):
            return type(
                "Output", (), {"images": [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))]}
            )()

    engine = LocalDiffusionEngine(
        "tiny",
        ModelDefinition(path="tiny", modality="video"),
        "local",
        pipeline=SingleFramePipeline(),
    )
    with pytest.raises(ModelLoadError, match="returned 1 frames instead of 2"):
        engine.generate_video(
            VideoGenerationRequest(
                model="tiny", prompt="waves", size="8x8", frames=2, steps=1, seed=2
            ),
            request_id="video-missing-frame",
            security_mode="strict",
            cancelled=threading.Event(),
            timeout=60,
        )


def test_public_sdk_image_and_video_methods_use_real_engines() -> None:
    from Xurcroh import Xurcroh

    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={"tiny": {"path": "tiny", "modality": "image"}},
    ) as client:
        client._runtime._engines["tiny"] = LocalDiffusionEngine(
            "tiny",
            ModelDefinition(path="tiny", modality="image"),
            "local",
            pipeline=TinyNeuralPipeline(),
        )
        response = client.images.generate(model="tiny", prompt="city", size="8x8", steps=1)
        assert response.execution.engine == "diffusers"
        assert base64.b64decode(response.data[0].b64_json).startswith(b"\x89PNG")

    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={"tiny": {"path": "tiny", "modality": "video"}},
    ) as client:
        client._runtime._engines["tiny"] = LocalDiffusionEngine(
            "tiny",
            ModelDefinition(path="tiny", modality="video"),
            "local",
            pipeline=TinyNeuralPipeline(),
        )
        response = client.videos.generate(model="tiny", prompt="waves", size="8x8", steps=1)
        assert response.execution.engine == "diffusers"
        assert b"ftyp" in base64.b64decode(response.data[0].b64_json)[:32]
