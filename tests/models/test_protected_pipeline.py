from __future__ import annotations

import torch

from models.test_diffusion_engine import TinyNeuralPipeline
from xurcroh.algebra.orthogonal import sample_orthogonal
from xurcroh.runtime.protected_diffusion import ProtectedDiffusionPipeline


def test_protected_pipeline_matches_unprotected_neural_denoising() -> None:
    generator = torch.Generator().manual_seed(301)
    initial = torch.randn(1, 4, 8, 8, generator=generator)
    basis = sample_orthogonal(4, seed=303, dtype=torch.float32)
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(305))
    torch.manual_seed(311)
    original = TinyNeuralPipeline()
    torch.manual_seed(311)
    protected = ProtectedDiffusionPipeline(TinyNeuralPipeline(), basis, permutation)
    protected_initial = protected.protect(initial)
    assert not torch.equal(protected_initial, initial)
    torch.testing.assert_close(protected.recover(protected_initial), initial)

    parameters = {
        "prompt": "city",
        "image": None,
        "width": 8,
        "height": 8,
        "num_inference_steps": 2,
        "generator": torch.Generator().manual_seed(307),
    }
    torch.manual_seed(309)
    expected = original(latents=initial, **parameters).images[0]
    torch.manual_seed(309)
    result = protected.generate(latents=protected_initial, **parameters)
    actual = result.images[0]
    assert actual.size == expected.size
    torch.testing.assert_close(
        protected.pipeline.final_latent, original.final_latent, atol=2e-3, rtol=2e-3
    )
    assert protected.pipeline.received_latent is not None
    torch.testing.assert_close(protected.recover(protected.pipeline.received_latent), initial)
    assert protected.scheduler_steps == 2
    assert protected.model_calls == 2


def test_local_diffusion_engine_supports_explicit_latent_protection() -> None:
    import base64
    import threading

    from xurcroh.config import ModelDefinition
    from xurcroh.requests import ImageGenerationRequest
    from xurcroh.runtime.diffusion import LocalDiffusionEngine

    engine = LocalDiffusionEngine(
        "tiny",
        ModelDefinition(path="tiny", modality="image", protect_latents=True),
        "local",
        pipeline=TinyNeuralPipeline(),
    )
    response = engine.generate_image(
        ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=2, seed=313),
        request_id="protected-image",
        security_mode="strict",
        cancelled=threading.Event(),
        timeout=60,
    )
    assert base64.b64decode(response.data[0].b64_json).startswith(b"\x89PNG")
    assert engine.protected_scheduler_steps == 2
    assert engine.protected_model_calls == 2
