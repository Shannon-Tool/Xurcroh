from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from xurcroh.errors import ModelLoadError

_DENOISERS = (
    "transformer",
    "unet",
    "transformer_2",
    "transformer_3",
    "high_noise_transformer",
    "low_noise_transformer",
)


@dataclass(frozen=True)
class MediaArchitectureAdapter:
    components: tuple[str, ...]
    modality: str
    workflows: frozenset[str]

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, modality: str) -> MediaArchitectureAdapter:
        if modality not in {"image", "video"}:
            raise ValueError(f"unsupported media modality: {modality}")
        root = Path(checkpoint)
        components = tuple(name for name in _DENOISERS if (root / name).is_dir())
        if not components:
            raise ModelLoadError(f"checkpoint contains no supported denoiser component: {root}")
        workflows = {"text_to_image"} if modality == "image" else {"text_to_video"}
        workflows.add("image_to_image" if modality == "image" else "image_to_video")
        return cls(components, modality, frozenset(workflows))

    @classmethod
    def from_pipeline(cls, pipeline: object, modality: str) -> MediaArchitectureAdapter:
        # Pipelines register optional denoisers they do not have as ``None``.
        components = tuple(name for name in _DENOISERS if getattr(pipeline, name, None) is not None)
        if not components:
            raise ModelLoadError("pipeline contains no supported denoiser component")
        workflows = {"text_to_image"} if modality == "image" else {"text_to_video"}
        parameters = inspect.signature(cast(Any, pipeline)).parameters
        if "image" in parameters:
            workflows.add("image_to_image" if modality == "image" else "image_to_video")
        return cls(components, modality, frozenset(workflows))
