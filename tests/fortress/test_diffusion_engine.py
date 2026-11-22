from __future__ import annotations

import threading
from pathlib import Path

import pytest

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.fortress.diffusion_engine import FortressDiffusionEngine
from xurcroh.fortress.module import FortressLinear
from xurcroh.requests import ImageGenerationRequest


def test_real_diffusers_pipeline_executes_with_fortress_linears(tmp_path: Path) -> None:
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    pipeline = diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler())
    pipeline.save_pretrained(tmp_path)

    engine = FortressDiffusionEngine(
        "tiny-ddpm",
        ModelDefinition(path=str(tmp_path), engine="fortress", modality="image"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    try:
        response = engine.generate_image(
            ImageGenerationRequest(
                model="tiny-ddpm", prompt="a small red square", size="64x64", steps=1, seed=11
            ),
            request_id="diffusion-test",
            security_mode="development",
            cancelled=threading.Event(),
            timeout=60,
        )
        assert response.data[0].b64_json
        assert any(
            isinstance(module, FortressLinear)
            for module in engine._boundary.pipeline.unet.modules()
        )
    finally:
        engine.close()


@pytest.mark.parametrize("shares", [2, 3])
def test_public_sdk_generates_image_with_fortress_diffusion(tmp_path: Path, shares: int) -> None:
    from Xurcroh import Xurcroh

    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    pipeline = diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler())
    pipeline.save_pretrained(tmp_path)
    with Xurcroh(
        model_path={
            "tiny-ddpm": {
                "path": str(tmp_path),
                "engine": "fortress",
                "fortress_shares": shares,
            }
        }
    ) as client:
        response = client.images.generate(
            model="tiny-ddpm", prompt="a small red square", size="64x64", steps=1, seed=11
        )
        unseeded = client.images.generate(
            model="tiny-ddpm", prompt="a small red square", size="64x64", steps=1
        )
        replay = client.images.generate(
            model="tiny-ddpm",
            prompt="a small red square",
            size="64x64",
            steps=1,
            seed=unseeded.execution.seed,
        )
        from xurcroh.receipts import TraceError, verify_trace

        trace = client.tasks.trace(unseeded.execution.request_id)
        public_key = client.tasks.public_key
    assert trace["request"]["kind"] == "image"
    assert trace["request"]["settings"] == {
        "model": "tiny-ddpm",
        "seed": None,
        "size": "64x64",
        "steps": 1,
        "strength": 0.75,
        "timeout": None,
    }
    assert trace["request"]["effective"] == {"seed": unseeded.execution.seed}
    assert trace["model"]["engine"] == "fortress-diffusion-local"
    assert "red square" not in str(trace)
    inputs = {"prompt": "a small red square", "image": None, "mask": None}
    assert verify_trace(
        trace, public_key=public_key, media_input=inputs, output=unseeded.data[0].b64_json
    ) == {"signature": True, "prompt": True, "output": True, "worker_checkpoints": None}
    # A media trace carries the same sharing and transport facts as a chat trace.
    assert trace["model"]["share_count"] >= 2
    assert trace["transport"]["worker_checkpoints"] is None
    with pytest.raises(TraceError, match="prompt"):
        verify_trace(trace, public_key=public_key, media_input={**inputs, "prompt": "other"})
    with pytest.raises(TraceError, match="output"):
        verify_trace(trace, public_key=public_key, output=response.data[0].b64_json + "A")
    assert response.data[0].b64_json
    assert response.execution.engine == "fortress-diffusion-local"
    assert response.execution.seed == 11
    assert unseeded.execution.seed is not None
    assert replay.data[0].b64_json == unseeded.data[0].b64_json


def test_video_pipeline_runs_inside_request_execution_deadline() -> None:
    import time
    from types import SimpleNamespace

    from xurcroh.errors import RequestTimeoutError
    from xurcroh.fortress.transport import check_execution_health
    from xurcroh.requests import VideoGenerationRequest

    class SlowPipeline:
        def __call__(self, **parameters):
            del parameters
            time.sleep(0.03)
            check_execution_health()
            return SimpleNamespace(frames=[object()])

    engine = FortressDiffusionEngine.__new__(FortressDiffusionEngine)
    engine._load = lambda: None
    engine._boundary = SimpleNamespace(
        pipeline=SlowPipeline(),
        decode_video=lambda frames, request: b"video",
    )
    engine._seed = lambda request: 1
    engine.worker_ids = ["local"]
    engine.engine_name = "fortress-diffusion-local"
    request = VideoGenerationRequest(
        model="tiny-video",
        prompt="waves",
        size="32x32",
        frames=2,
        fps=2,
        steps=1,
    )
    with pytest.raises(RequestTimeoutError, match="deadline"):
        engine.generate_video(
            request,
            request_id="video-timeout",
            security_mode="development",
            cancelled=threading.Event(),
            timeout=0.01,
        )
