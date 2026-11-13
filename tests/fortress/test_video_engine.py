from __future__ import annotations

import threading
from pathlib import Path

import pytest

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.fortress.diffusion_engine import FortressDiffusionEngine
from xurcroh.fortress.module import FortressLinear
from xurcroh.requests import VideoGenerationRequest


def test_real_video_pipeline_generates_frames_with_fortress(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    unet = diffusers.UNet3DConditionModel(
        sample_size=2,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock3D",),
        up_block_types=("UpBlock3D",),
        block_out_channels=(4,),
        layers_per_block=1,
        norm_num_groups=4,
        cross_attention_dim=16,
        attention_head_dim=4,
    )
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=4,
    )
    config = transformers.CLIPTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        projection_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    text_encoder = transformers.CLIPTextModel(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_llama_path)
    tokenizer.model_max_length = 8
    pipeline = diffusers.TextToVideoSDPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=diffusers.DDIMScheduler(),
    )
    pipeline.save_pretrained(tmp_path)
    engine = FortressDiffusionEngine(
        "tiny-video",
        ModelDefinition(path=str(tmp_path), engine="fortress", modality="video"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    try:
        response = engine.generate_video(
            VideoGenerationRequest(
                model="tiny-video",
                prompt="ocean waves",
                size="32x32",
                frames=2,
                fps=2,
                steps=1,
                seed=12,
            ),
            request_id="video-test",
            security_mode="development",
            cancelled=threading.Event(),
            timeout=180,
        )
        assert response.data[0].b64_json
        assert any(
            isinstance(module, FortressLinear)
            for module in engine._boundary.pipeline.unet.modules()
        )
    finally:
        engine.close()

    from Xurcroh import Xurcroh

    with Xurcroh(
        model_path={
            "tiny-video": {
                "path": str(tmp_path),
                "engine": "fortress",
                "modality": "video",
            }
        }
    ) as client:
        public_response = client.videos.generate(
            model="tiny-video",
            prompt="ocean waves",
            size="32x32",
            frames=2,
            fps=2,
            steps=1,
            seed=12,
            timeout=180,
        )
    assert public_response.data[0].b64_json
    assert public_response.execution.engine == "fortress-diffusion-local"
