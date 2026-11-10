from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import torch
import uvicorn

from models.test_diffusion_engine import TinyNeuralPipeline
from Xurcroh import Xurcroh
from xurcroh.config import ClusterConfig, ModelDefinition, ModelRegistry
from xurcroh.runtime import RuntimeService
from xurcroh.runtime.diffusion import LocalDiffusionEngine
from xurcroh.runtime.media_boundary import DiffusionMediaBoundary
from xurcroh.worker import create_worker_app


class TinyMediaBoundary(DiffusionMediaBoundary):
    def __init__(self) -> None:
        super().__init__(TinyNeuralPipeline())

    def prepare_video(self, request) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        from xurcroh.requests import ImageGenerationRequest

        image_request = ImageGenerationRequest(
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            steps=request.steps,
            seed=request.seed,
        )
        tensors, parameters = self.prepare(image_request)
        parameters.update({"num_frames": request.frames, "fps": request.fps})
        return tensors, parameters

    def decode_video(self, latent: torch.Tensor, request) -> bytes:
        import imageio.v2 as imageio

        with torch.inference_mode():
            decoded = self.pipeline.decode_latents(latent)
            pixels = self.pipeline.image_processor.postprocess(decoded, output_type="np")[0]
        pixels = (pixels * 255).round().astype("uint8")
        buffer = io.BytesIO()
        with imageio.get_writer(buffer, format="mp4", fps=request.fps, codec="libx264") as writer:
            for _ in range(request.frames):
                writer.append_data(pixels)
        return buffer.getvalue()


def test_protected_remote_image_and_video_use_client_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    registry = ModelRegistry.normalize(
        {
            "image": {"path": "tiny-image", "modality": "image"},
            "video": {"path": "tiny-video", "modality": "video"},
        }
    )
    runtime = RuntimeService(ClusterConfig.normalize({"endpoint": "local://cpu"}), registry)
    runtime._engines["image"] = LocalDiffusionEngine(
        "image",
        ModelDefinition(path="tiny-image", modality="image"),
        "media-worker",
        pipeline=TinyNeuralPipeline(),
    )
    runtime._engines["video"] = LocalDiffusionEngine(
        "video",
        ModelDefinition(path="tiny-video", modality="video"),
        "media-worker",
        pipeline=TinyNeuralPipeline(),
    )
    app = create_worker_app(registry, worker_id="media-worker", runtime=runtime)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        if time.monotonic() - started > 10:
            raise RuntimeError("protected media worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    monkeypatch.setattr(
        "xurcroh.runtime.media_boundary.DiffusionMediaBoundary.from_definition",
        lambda definition: TinyMediaBoundary(),
    )
    try:
        with Xurcroh(
            cluster={
                "endpoint": f"http+dev://127.0.0.1:{port}",
                "security": {"mode": "development"},
            },
            model_path={
                "image": {
                    "path": "tiny-image",
                    "modality": "image",
                    "protect_latents": True,
                },
                "video": {
                    "path": "tiny-video",
                    "modality": "video",
                    "protect_latents": True,
                },
            },
        ) as client:
            image = client.images.generate(
                model="image", prompt="city", size="8x8", steps=2, seed=411
            )
            video = client.videos.generate(
                model="video", prompt="waves", size="8x8", frames=2, fps=2, steps=2, seed=413
            )
            from models.test_vlm_runtime import png_data_url

            edited = client.images.edit(
                model="image", prompt="city", image=png_data_url(), size="8x8", steps=2, seed=415
            )
        assert image.execution.engine == "protected-remote-diffusers"
        assert base64.b64decode(image.data[0].b64_json).startswith(b"\x89PNG")
        assert base64.b64decode(edited.data[0].b64_json).startswith(b"\x89PNG")
        assert video.execution.engine == "protected-remote-diffusers"
        assert b"ftyp" in base64.b64decode(video.data[0].b64_json)[:32]
        assert app.state.protected_media_sessions.active_count == 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
