from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import torch

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ModelLoadError, ProtocolError
from xurcroh.kem import HybridKemKey, open_sealed
from xurcroh.runtime.media_sessions import (
    ProtectedMediaSessionStore,
    encode_public,
    media_session_info,
    open_tensors,
    seal_tensors,
)


def _client_master(
    private: HybridKemKey, session_id: str, encapsulated: str, handshake: str
) -> bytes:
    return open_sealed(
        private,
        media_session_info(session_id),
        base64.b64decode(encapsulated) + base64.b64decode(handshake),
    )


def test_media_session_authenticates_tensors_and_rejects_replay() -> None:
    client_private = HybridKemKey.generate()
    store = ProtectedMediaSessionStore()
    session_id, encapsulated, handshake = store.create(encode_public(client_private.public_key))
    master = _client_master(client_private, session_id, encapsulated, handshake)
    sender = PipelineTensorInterceptor(master, b"client", b"worker", session_id.encode())
    session = store.get(session_id)
    packet = seal_tensors({"value": torch.tensor([1, 2, 3])}, "request-1", sender)
    assert open_tensors(packet, "request-1", session.receiver)["value"].tolist() == [1, 2, 3]
    with pytest.raises(ProtocolError, match="replay"):
        open_tensors(packet, "request-1", session.receiver)
    with pytest.raises(ProtocolError):
        open_tensors(packet[:-2] + "AA", "request-1", session.receiver)
    store.remove(session_id)
    assert store.active_count == 0
    with pytest.raises(ProtocolError, match="expired"):
        store.get(session_id)


def test_media_session_capacity_fails_closed() -> None:
    store = ProtectedMediaSessionStore(capacity=1)
    private = HybridKemKey.generate()
    store.create(encode_public(private.public_key))
    with pytest.raises(ProtocolError, match="capacity"):
        store.create(encode_public(private.public_key))
    store.close()


def test_media_session_directions_use_distinct_keys_for_the_same_sequence() -> None:
    client_private = HybridKemKey.generate()
    store = ProtectedMediaSessionStore()
    session_id, encapsulated, handshake = store.create(encode_public(client_private.public_key))
    master = _client_master(client_private, session_id, encapsulated, handshake)
    context = session_id.encode()
    client_sender = PipelineTensorInterceptor(master, b"client", b"worker", context)
    client_receiver = PipelineTensorInterceptor(master, b"client", b"worker", context)
    session = store.get(session_id)

    request = seal_tensors({"value": torch.tensor([1])}, "shared-request", client_sender)
    response = seal_tensors({"value": torch.tensor([1])}, "shared-request", session.sender)

    assert request != response
    assert open_tensors(request, "shared-request", session.receiver)["value"].item() == 1
    assert open_tensors(response, "shared-request", client_receiver)["value"].item() == 1
    with pytest.raises(ProtocolError, match="authentication"):
        open_tensors(response, "shared-request", session.receiver)
    with pytest.raises(ProtocolError, match="authentication"):
        open_tensors(request, "shared-request", client_receiver)
    store.close()


def _assert_no_seed_material_in_parameters(
    tensors: dict[str, torch.Tensor], parameters: dict[str, object]
) -> None:
    from xurcroh.algebra.orthogonal import sample_orthogonal

    assert "generator_seed" not in parameters
    assert "generator_seed" in tensors
    basis = tensors["basis"]
    permutation = tensors["permutation"]
    channels = basis.shape[0]
    positions = permutation.numel()
    for value in parameters.values():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        candidate_basis = sample_orthogonal(channels, seed=value + 10007, dtype=torch.float32)
        assert not torch.equal(candidate_basis, basis)
        candidate_permutation = torch.randperm(
            positions, generator=torch.Generator().manual_seed(value + 10009)
        )
        assert not torch.equal(candidate_permutation, permutation)


def test_prepare_seals_generator_seed_and_parameters_reveal_no_transform_secret() -> None:
    from models.test_diffusion_engine import TinyNeuralPipeline
    from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
    from xurcroh.runtime.media_boundary import DiffusionMediaBoundary

    boundary = DiffusionMediaBoundary(TinyNeuralPipeline())
    image = ImageGenerationRequest(model="tiny", prompt="city", size="8x8", steps=2, seed=411)
    tensors, parameters = boundary.prepare(image)
    _assert_no_seed_material_in_parameters(tensors, parameters)

    video = VideoGenerationRequest(
        model="tiny", prompt="waves", size="8x8", frames=2, fps=2, steps=2, seed=413
    )
    video_tensors, video_parameters = boundary.prepare_video(video)
    _assert_no_seed_material_in_parameters(video_tensors, video_parameters)


def test_safetensors_weight_map_traversal_fails_closed(tmp_path: Path) -> None:
    from xurcroh.formats.safetensors import SafetensorsReader

    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "tiny"}')
    index_path = root / "model.safetensors.index.json"
    for filename in ("../outside.safetensors", "/etc/xurcroh-escape.safetensors"):
        index_path.write_text(json.dumps({"weight_map": {"weight": filename}}))
        with pytest.raises(ModelLoadError, match="escapes"):
            SafetensorsReader(root)
