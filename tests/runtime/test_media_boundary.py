from __future__ import annotations

import numpy as np
import pytest
import torch

from integration.test_protected_media import TinyMediaBoundary
from models.test_diffusion_engine import TinyNeuralPipeline
from models.test_vlm_runtime import png_data_url
from xurcroh.errors import CapabilityError
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime.media_boundary import DiffusionMediaBoundary
from xurcroh.runtime.protected_diffusion import ProtectedDiffusionPipeline


def test_raw_media_inputs_and_real_decoders() -> None:
    boundary = TinyMediaBoundary()
    image_request = ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=2, seed=1)
    initial, conditioning, parameters = boundary.raw_inputs(image_request)
    assert initial.shape == (1, 4, 8, 8)
    assert conditioning.shape[0] == 1
    assert parameters["num_inference_steps"] == 2

    video_request = VideoGenerationRequest(
        model="tiny", prompt="waves", size="8x8", frames=2, fps=2, steps=2, seed=2
    )
    video_latent, video_conditioning, video_parameters = boundary.raw_video_inputs(video_request)
    assert video_latent.shape == (1, 4, 2, 8, 8)
    assert video_conditioning.shape[0] == 1
    assert video_parameters["num_frames"] == 2

    png = boundary.decode(initial, image_request)
    assert png.startswith(b"\x89PNG")
    video = boundary.decode_video(video_latent[:, :, 0], video_request)
    assert b"ftyp" in video[:32]


def test_source_latent_editing_and_prepare_shapes() -> None:
    boundary = TinyMediaBoundary()
    request = ImageGenerationRequest(
        model="tiny",
        prompt="city",
        image=png_data_url(),
        size="8x8",
        steps=2,
        seed=3,
        strength=0.4,
    )
    initial, _, _ = boundary.raw_inputs(request)
    assert initial.shape == (1, 4, 8, 8)
    tensors, parameters = boundary.prepare(request)
    assert tensors["latents"].shape == initial.shape
    assert tensors["basis"].shape == (4, 4)
    assert tensors["permutation"].numel() == 64
    assert parameters["strength"] == 0.4


def test_media_boundary_rejects_missing_decoders_and_empty_results() -> None:
    with pytest.raises(CapabilityError, match="UNet client boundary"):
        DiffusionMediaBoundary(object()).raw_inputs(
            ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=1)
        )

    class EmptyPipeline:
        def decode_latents(self, latent):
            return latent

        @property
        def image_processor(self):
            return type("Processor", (), {"postprocess": lambda self, pixels, output_type: []})()

    with pytest.raises(CapabilityError, match="returned no images"):
        DiffusionMediaBoundary(EmptyPipeline()).decode(
            torch.zeros(1, 4, 8, 8),
            ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=1),
        )

    class InvalidVideoPipeline:
        def decode_latents(self, latent):
            return np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(CapabilityError, match="RGB frames"):
        DiffusionMediaBoundary(InvalidVideoPipeline()).decode_video(
            torch.zeros(1, 4, 1, 8, 8),
            VideoGenerationRequest(model="tiny", prompt="waves", size="8x8", steps=1),
        )


def test_transport_prepare_video_uses_protected_latents() -> None:
    boundary = TinyMediaBoundary()
    tensors, parameters = boundary.prepare_video(
        VideoGenerationRequest(
            model="tiny", prompt="waves", size="8x8", frames=2, fps=2, steps=2, seed=4
        )
    )
    assert tensors["latents"].shape == (1, 4, 8, 8)
    assert parameters["fps"] == 2


def test_video_preparation_uses_spacetime_permutation() -> None:
    boundary = DiffusionMediaBoundary(TinyNeuralPipeline())
    request = VideoGenerationRequest(
        model="tiny", prompt="waves", size="8x8", frames=2, fps=7, steps=2, seed=23
    )
    tensors, parameters = boundary.prepare_video(request)
    assert tensors["latents"].shape == (1, 4, 2, 8, 8)
    assert tensors["permutation"].numel() == 128
    assert parameters["fps"] == 7
    protected = ProtectedDiffusionPipeline(
        boundary.pipeline, tensors["basis"], tensors["permutation"]
    )
    original, _, _ = boundary.raw_video_inputs(request)
    torch.testing.assert_close(protected.recover(tensors["latents"]), original)


def test_vae_decode_and_from_definition_paths(monkeypatch, tmp_path) -> None:
    from xurcroh.config import ModelDefinition

    boundary = DiffusionMediaBoundary(TinyNeuralPipeline())
    monkeypatch.delattr(TinyNeuralPipeline, "decode_latents")
    latent = torch.randn(1, 4, 8, 8)
    request = ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=1)
    assert boundary.decode(latent, request).startswith(b"\x89PNG")

    captured = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return TinyNeuralPipeline()

    monkeypatch.setattr("diffusers.DiffusionPipeline", Pipeline)
    loaded = DiffusionMediaBoundary.from_definition(
        ModelDefinition(path=str(tmp_path), modality="image", revision="test-revision")
    )
    assert isinstance(loaded.pipeline, TinyNeuralPipeline)
    assert captured == {
        "path": str(tmp_path),
        "local_files_only": True,
        "trust_remote_code": False,
        "revision": "test-revision",
    }


def test_media_decoder_failure_paths() -> None:
    class NoVideoDecoder:
        unet = TinyNeuralPipeline().unet
        encode_prompt = TinyNeuralPipeline().encode_prompt

    boundary = DiffusionMediaBoundary(NoVideoDecoder())
    with pytest.raises(CapabilityError, match="latent video decoder"):
        boundary.decode_video(
            torch.zeros(1, 4, 1, 8, 8),
            VideoGenerationRequest(model="tiny", prompt="waves", size="8x8", steps=1),
        )

    class NoImageDecoder:
        pass

    with pytest.raises(CapabilityError, match="latent image decoder"):
        DiffusionMediaBoundary(NoImageDecoder()).decode(
            torch.zeros(1, 4, 8, 8),
            ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=1),
        )
