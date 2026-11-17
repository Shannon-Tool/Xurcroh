from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.fortress.diffusion_engine import FortressDiffusionEngine
from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest


class _Pipeline:
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}
        self.vae = object()

    def __call__(self, **parameters: Any) -> Any:
        self.parameters = parameters
        return type("Result", (), {"frames": torch.zeros(1, 1, 1, 1, 1, 1)})()


class _Boundary:
    def __init__(self) -> None:
        self.pipeline = _Pipeline()

    def decode_video(self, frames: torch.Tensor, request: VideoGenerationRequest) -> bytes:
        return b"video"


def _image() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_image_to_video_passes_client_image_to_pipeline(tmp_path: Path) -> None:
    engine = FortressDiffusionEngine(
        "video",
        ModelDefinition(path=str(tmp_path), modality="video", engine="fortress"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    boundary = _Boundary()
    engine._boundary = boundary  # type: ignore[assignment]
    response = engine.generate_video(
        VideoGenerationRequest(model="video", prompt="waves", image=_image(), steps=1, seed=4),
        request_id="i2v",
        security_mode="development",
        cancelled=threading.Event(),
        timeout=5,
    )
    assert response.data[0].b64_json == "dmlkZW8="
    assert isinstance(boundary.pipeline.parameters["image"], Image.Image)
    engine.close()


def test_image_edit_mask_is_validated_and_forwarded(tmp_path: Path) -> None:
    class ImagePipeline:
        def __init__(self) -> None:
            self.parameters: dict[str, Any] = {}
            self.vae = object()

        def __call__(self, **parameters: Any) -> Any:
            self.parameters = parameters
            return type("Result", (), {"images": torch.zeros(1, 1, 1, 1)})()

    class ImageBoundary:
        def __init__(self) -> None:
            self.pipeline = ImagePipeline()

        def decode(self, latent: torch.Tensor, request: ImageGenerationRequest) -> bytes:
            return b"image"

    engine = FortressDiffusionEngine(
        "image",
        ModelDefinition(path=str(tmp_path), modality="image", engine="fortress"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    boundary = ImageBoundary()
    engine._boundary = boundary  # type: ignore[assignment]
    response = engine.generate_image(
        ImageGenerationRequest(
            model="image", prompt="edit", image=_image(), mask=_image(), steps=1, seed=4
        ),
        request_id="edit",
        security_mode="development",
        cancelled=threading.Event(),
        timeout=5,
    )
    assert response.data[0].b64_json == "aW1hZ2U="
    assert isinstance(boundary.pipeline.parameters["mask_image"], Image.Image)
    engine.close()
