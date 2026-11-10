from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from xurcroh.errors import CapabilityError, ShardIntegrityError
from xurcroh.models.dit import reference_dit_stage
from xurcroh.shards.dit_compiler import weights_from_tensors
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key, verify_manifest_digest


def _document(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        document = json.loads(manifest_path.read_text())
        if document["format"] != "xurcroh-dit-manifest" or document["version"] != 1:
            raise ShardIntegrityError("unsupported DiT manifest")
        verify_manifest_digest(document)
        if document["stage_count"] != 1 or len(document["stages"]) != 1:
            raise ShardIntegrityError("DiT manifest requires exactly one stage")
        return cast(dict[str, Any], document)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ShardIntegrityError("invalid DiT manifest") from error


class CompiledDiTStage:
    def __init__(self, manifest_path: str | Path, private_key: X25519PrivateKey) -> None:
        path = Path(manifest_path)
        document = _document(path)
        stage = document["stages"][0]
        public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        if public.hex() != stage["recipient_public"]:
            raise ShardIntegrityError("DiT recipient identity mismatch")
        envelope = json.loads((path.parent / stage["key_envelope"]).read_text())
        key = open_stage_key(
            envelope,
            private_key,
            manifest_digest=document["digest"],
            checkpoint_fingerprint=document["checkpoint_fingerprint"],
        )
        self._tensors, _ = read_shard(path.parent / stage["file"], key)
        self.weights = weights_from_tensors(self._tensors)
        self.closed = False

    def generate(
        self,
        parameters: dict[str, Any],
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("compiled DiT stage is closed")
        steps = int(parameters.get("num_inference_steps", 1))
        if steps < 1 or steps >= len(self.weights.timesteps):
            raise ValueError("DiT steps exceed the configured schedule")
        return reference_dit_stage(self.weights, latents, prompt_embeds, steps=steps)

    def close(self) -> None:
        if not self.closed:
            for tensor in self._tensors.values():
                tensor.zero_()
            self._tensors.clear()
            self.closed = True


class CompiledDiTClientBoundary:
    def __init__(self, manifest_path: str | Path, boundary_key: bytes) -> None:
        path = Path(manifest_path)
        document = _document(path)
        self._tensors, _ = read_shard(path.parent / document["boundary"]["file"], boundary_key)
        self.closed = False

    def prepare(self, latent: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.closed:
            raise RuntimeError("compiled DiT client boundary is closed")
        return {
            "latents": latent[self._tensors["latent_permutation"]]
            @ self._tensors["latent_basis"].T,
            "prompt_embeds": text @ self._tensors["text_basis"].T,
        }

    def recover(self, latent: torch.Tensor) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("compiled DiT client boundary is closed")
        inverse = torch.argsort(self._tensors["latent_permutation"])
        return (latent @ self._tensors["latent_basis"])[inverse]

    def close(self) -> None:
        if not self.closed:
            for tensor in self._tensors.values():
                tensor.zero_()
            self._tensors.clear()
            self.closed = True


class CompiledDiTStageEngine:
    def __init__(self, name: str, stage: CompiledDiTStage, worker_id: str, modality: str) -> None:
        self.name = name
        self.stage = stage
        self.worker_id = worker_id
        self.worker_ids = [worker_id]
        self.modality = modality
        self.engine_name = "compiled-dit-stage"

    def require_modality(self, expected: str) -> None:
        if expected != self.modality:
            raise ValueError(f"compiled DiT stage only supports {self.modality}")

    def generate(
        self,
        parameters: dict[str, Any],
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.stage.generate(
            parameters,
            latents=latents,
            prompt_embeds=prompt_embeds,
        )

    def generate_image(self, request: Any, **kwargs: Any) -> Any:
        raise CapabilityError("compiled DiT workers accept requests through protected sessions")

    def generate_video(self, request: Any, **kwargs: Any) -> Any:
        raise CapabilityError("compiled DiT workers accept requests through protected sessions")

    def close(self) -> None:
        self.stage.close()
