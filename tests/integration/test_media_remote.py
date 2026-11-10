from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import uvicorn

from models.test_diffusion_engine import TinyNeuralPipeline
from Xurcroh import Xurcroh
from xurcroh.config import ClusterConfig, ModelDefinition, ModelRegistry
from xurcroh.runtime import RuntimeService
from xurcroh.runtime.diffusion import LocalDiffusionEngine
from xurcroh.worker import create_worker_app


def test_remote_image_and_video_generation_use_real_worker_models(tmp_path: Path) -> None:
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
            raise RuntimeError("media worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        with Xurcroh(
            cluster={
                "endpoint": f"http+dev://127.0.0.1:{port}",
                "security": {"mode": "development"},
            },
            model_path={
                "image": {"path": "remote-image", "modality": "image"},
                "video": {"path": "remote-video", "modality": "video"},
            },
        ) as client:
            image = client.images.generate(
                model="image", prompt="city", size="8x8", steps=1, seed=1
            )
            video = client.videos.generate(
                model="video", prompt="waves", size="8x8", frames=2, fps=2, steps=1, seed=2
            )
        assert base64.b64decode(image.data[0].b64_json).startswith(b"\x89PNG")
        assert image.execution.engine == "remote-diffusers"
        assert base64.b64decode(video.data[0].b64_json)[:32].find(b"ftyp") >= 0
        assert video.execution.engine == "remote-diffusers"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
