from __future__ import annotations

from dataclasses import dataclass

import torch

from xurcroh.algebra.rmsnorm import homogenize_embedding_norms, unit_rms_norm


class GemmaVisionEmbedder:
    def __init__(self, projection: torch.Tensor, *, epsilon: float | None = None) -> None:
        if projection.ndim != 2 or projection.shape[1] != 48 * 48 * 3:
            raise ValueError("Gemma vision projection must have 6912 input channels")
        self.projection = projection
        self.epsilon = epsilon

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have channel, height, and width dimensions")
        channels, height, width = image.shape
        if height % 48 or width % 48:
            raise ValueError("Gemma image dimensions must be divisible by 48")
        grid_height, grid_width = height // 48, width // 48
        patches = (
            image.reshape(channels, grid_height, 48, grid_width, 48)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_height * grid_width, channels * 48 * 48)
            .to(self.projection.dtype)
        )
        if self.epsilon is not None:
            patches = torch.nn.functional.layer_norm(
                patches, (patches.shape[-1],), eps=self.epsilon
            )
        output = patches @ self.projection.T
        if self.epsilon is not None:
            output = torch.nn.functional.layer_norm(output, (output.shape[-1],), eps=self.epsilon)
        return output


class GemmaAudioEmbedder:
    def __init__(self, projection: torch.Tensor, *, epsilon: float = 1e-6) -> None:
        if projection.ndim != 2 or projection.shape[1] != 640:
            raise ValueError("Gemma audio projection must have 640 input channels")
        self.projection = projection
        self.epsilon = epsilon

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("audio waveform must be a nonempty vector")
        frames = (waveform.numel() + 639) // 640
        padded = torch.nn.functional.pad(waveform, (0, frames * 640 - waveform.numel()))
        padded = padded.to(self.projection.dtype)
        return unit_rms_norm(padded.reshape(frames, 640), self.epsilon) @ self.projection.T


def pad_with_chaff(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    bucket_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.shape[0] != attention_mask.shape[0] or bucket_size < embeddings.shape[0]:
        raise ValueError("chaff bucket must contain every real token")
    if not torch.all(attention_mask):
        raise ValueError("real attention mask must precede chaff padding")
    generator = torch.Generator().manual_seed(seed)
    chaff = torch.randn(
        bucket_size - embeddings.shape[0],
        embeddings.shape[1],
        generator=generator,
        dtype=embeddings.dtype,
    ).to(embeddings.device)
    padded = torch.cat([embeddings, chaff], dim=0)
    mask = torch.cat(
        [attention_mask, torch.zeros(bucket_size - embeddings.shape[0], dtype=torch.bool)],
        dim=0,
    ).to(attention_mask.device)
    return padded, mask


@dataclass(frozen=True)
class BoundaryPacket:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


def prepare_boundary_packet(
    embeddings: torch.Tensor,
    *,
    bucket_size: int,
    seed: int,
) -> BoundaryPacket:
    homogenized = homogenize_embedding_norms(embeddings)
    mask = torch.ones(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    padded, padded_mask = pad_with_chaff(
        homogenized,
        mask,
        bucket_size=bucket_size,
        seed=seed,
    )
    return BoundaryPacket(padded, padded_mask)


class DeepStackInjector:
    def __init__(
        self,
        connectors: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        self.connectors = {
            layer: (connector, source_basis, target_basis)
            for layer, connector, source_basis, target_basis in connectors
        }

    def inject(
        self,
        hidden: torch.Tensor,
        features: dict[int, torch.Tensor],
        *,
        target_layer: int,
    ) -> torch.Tensor:
        connector, _, target_basis = self.connectors[target_layer]
        transformed_connector = target_basis @ connector @ self.connectors[target_layer][1].T
        return hidden + features[target_layer] @ transformed_connector.T
