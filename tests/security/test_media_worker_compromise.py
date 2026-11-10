from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import pytest
import torch
import uvicorn

from models.test_dit_stage import tiny_dit_weights
from shards.identities import StageIdentities
from xurcroh.config import (
    ClusterConfig,
    ModelDefinition,
    ModelRegistry,
    SecurityConfig,
    WorkerConfig,
)
from xurcroh.errors import CapabilityError
from xurcroh.models.dit import reference_dit_stage
from xurcroh.requests import ImageGenerationRequest
from xurcroh.runtime.dit_stage import CompiledDiTStage, CompiledDiTStageEngine
from xurcroh.runtime.media_protected import RemoteProtectedMediaEngine
from xurcroh.runtime.service import RuntimeService
from xurcroh.shards.dit_compiler import DitCompiler
from xurcroh.worker import create_worker_app


class TinyCompiledBoundary:
    def __init__(self, latent: torch.Tensor, text: torch.Tensor) -> None:
        self.latent = latent
        self.text = text
        self.decoded: torch.Tensor | None = None

    def raw_inputs(self, request):
        return self.latent, self.text, {"num_inference_steps": 2}

    def decode(self, latent: torch.Tensor, request) -> bytes:
        self.decoded = latent.clone()
        return b"decoded"


def test_compiled_media_request_has_no_transformation_secrets_and_matches_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    original = tiny_dit_weights()
    identities = StageIdentities.create(1)
    compiled = DitCompiler().compile(
        original,
        tmp_path / "compiled",
        boundary_key=identities.boundary_key,
        recipient=identities.recipients[0],
        seed=409,
    )
    latent = torch.randn(3, 2, dtype=torch.float64)
    text = torch.randn(2, 4, dtype=torch.float64)
    boundary = TinyCompiledBoundary(latent, text)
    monkeypatch.setenv("TEST_XURCROH_CLIENT_BOUNDARY_KEY", identities.boundary_key.hex())
    monkeypatch.setattr(
        "xurcroh.runtime.media_boundary.DiffusionMediaBoundary.from_definition",
        lambda definition: boundary,
    )
    registry = ModelRegistry.normalize(
        {
            "image": {
                "path": "tiny-dit",
                "modality": "image",
                "manifest_path": str(compiled.manifest_path),
                "boundary_key_env": "TEST_XURCROH_CLIENT_BOUNDARY_KEY",
                "protect_latents": True,
            }
        }
    )
    runtime = RuntimeService(ClusterConfig.normalize({"endpoint": "local://cpu"}), registry)
    runtime._engines["image"] = CompiledDiTStageEngine(
        "image",
        CompiledDiTStage(compiled.manifest_path, identities.private_keys[0]),
        "dit-worker",
        "image",
    )
    app = create_worker_app(registry, worker_id="dit-worker", runtime=runtime)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        if time.monotonic() - started > 10:
            raise RuntimeError("compiled DiT worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]

    def stop_server() -> None:
        server.should_exit = True
        thread.join(timeout=10)

    request.addfinalizer(stop_server)
    definition = registry.get("image")
    engine = RemoteProtectedMediaEngine(
        "image",
        definition,
        [WorkerConfig(name="dit-worker", endpoint=f"http+dev://127.0.0.1:{port}")],
        SecurityConfig(mode="development"),
    )
    request_keys = []
    from xurcroh.runtime import media_sessions

    original_open = media_sessions.open_tensors

    def capture(packet, request_id, session):
        tensors = original_open(packet, request_id, session)
        request_keys.append(set(tensors))
        return tensors

    monkeypatch.setattr(media_sessions, "open_tensors", capture)
    response = engine.generate_image(
        request=ImageGenerationRequest(model="image", prompt="city", size="8x8", steps=2, seed=419),
        request_id="compiled-media",
        security_mode="development",
        cancelled=threading.Event(),
        timeout=60,
    )
    assert base64.b64decode(response.data[0].b64_json) == b"decoded"
    assert request_keys == [{"latents", "prompt_embeds"}]
    assert boundary.decoded is not None
    expected = reference_dit_stage(original, latent, text, steps=2)
    torch.testing.assert_close(boundary.decoded, expected, atol=3e-10, rtol=3e-10)
    engine.close()
    runtime.close()


def test_strict_media_rejects_worker_side_latent_recovery() -> None:
    with pytest.raises(CapabilityError, match="compiled equivariant DiT manifest"):
        RemoteProtectedMediaEngine(
            "image",
            ModelDefinition(path="tiny-unet", modality="image", protect_latents=True),
            [WorkerConfig(name="worker", endpoint="https://worker.invalid")],
            SecurityConfig(mode="strict"),
        )
