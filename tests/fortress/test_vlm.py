from __future__ import annotations

import base64
import io
import shutil
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from xurcroh.config import ModelDefinition, WorkerConfig
from xurcroh.fortress.engine import FortressEngine
from xurcroh.models.vlm import GemmaAudioEmbedder, GemmaVisionEmbedder
from xurcroh.requests import CompletionRequest


def test_vlm_boundary_projects_image_and_audio_locally(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "vlm"
    shutil.copytree(tiny_llama_path, model_path)
    tensors = {}
    with safe_open(model_path / "model.safetensors", framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            tensors[name] = handle.get_tensor(name)
    vision_projection = torch.randn(16, 6912, dtype=torch.float64) / 100
    audio_projection = torch.randn(16, 640, dtype=torch.float64) / 100
    tensors["vision_projection.weight"] = vision_projection
    tensors["audio_projection.weight"] = audio_projection
    save_file(tensors, str(model_path / "model.safetensors"))

    image_path = tmp_path / "red.png"
    Image.new("RGB", (48, 48), color=(255, 0, 0)).save(image_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.arange(800, dtype="<i2").tobytes())
    request = CompletionRequest.validated(
        model="vlm",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(buffer.getvalue()).decode(),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        max_tokens=1,
        temperature=0,
    )
    engine = FortressEngine(
        "vlm",
        ModelDefinition(path=str(model_path), engine="fortress", modality="vlm"),
        [WorkerConfig(name="local", endpoint="local://cpu")],
    )
    try:
        hidden, prompt_tokens = engine._prepare_hidden(request)
        text = engine._text_hidden("user: hello")
        pixels = (
            torch.from_numpy(np.array(Image.new("RGB", (48, 48), color=(255, 0, 0)))).permute(
                2, 0, 1
            )
            / 255
        )
        vision = GemmaVisionEmbedder(vision_projection).embed(pixels)
        audio = GemmaAudioEmbedder(audio_projection).embed(
            torch.arange(800, dtype=torch.float64) / 32768
        )
        assert prompt_tokens == text.shape[0] + vision.shape[0] + audio.shape[0]
        torch.testing.assert_close(hidden[: text.shape[0]], text)
        torch.testing.assert_close(hidden[text.shape[0] : text.shape[0] + 1], vision)
        torch.testing.assert_close(hidden[text.shape[0] + 1 :], audio)
        assert engine._compiled is not None
        assert "vision_projection.weight" not in engine._compiled.operations
        assert "audio_projection.weight" not in engine._compiled.operations
    finally:
        engine.close()
