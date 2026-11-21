#!/usr/bin/env python3
"""Fetch the pinned miniature Diffusers pipelines used by media theory tests.

The media ladder families (PixArt-Sigma, Flux, Qwen-Image, Wan) have public
miniature pipelines with the same component classes as the official
checkpoints. They are random-weight test fixtures: running them shows that a
rung's graph, placement and verification path execute, and nothing about the
official checkpoint's quality.

Usage:
    python scripts/fortress/fetch_tiny_pipelines.py /secure/tiny-pipelines
    XURCROH_TINY_PIPELINES=/secure/tiny-pipelines \\
        pytest -m slow tests/fortress/test_media_ladder_theory.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PIPELINES = {
    "pixart": (
        "hf-internal-testing/tiny-pixart-sigma-pipe",
        "57321596a551be525930e70b11f2d67460613a4a",
    ),
    "flux": ("hf-internal-testing/tiny-flux-pipe", "a98bc4ec6a80c47c477ed22d7e81c79872d9c723"),
    "qwenimage": (
        "hf-internal-testing/tiny-qwenimage-pipe",
        "4d1fcf7bbc41fff7f4e686dba11f871a3ad84a09",
    ),
    "wan": ("hf-internal-testing/tiny-wan-pipe", "8a97b5eb008dd10f4ac0d3f55d41f4acb3d65021"),
}


def _compose(root: Path) -> None:
    """Derive the fixtures that have no public miniature from the fetched ones."""
    import torch
    from diffusers import (
        AutoencoderKL,
        PixArtSigmaPipeline,
        PixArtTransformer2DModel,
        WanImageToVideoPipeline,
        WanPipeline,
        WanTransformer3DModel,
    )
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(20261121)

    # PixArt-Sigma bins the requested size by the transformer's sample size and
    # accepts only the official values; the public fixture uses 8. Rebuild the
    # denoiser and VAE at the 256-pixel configuration (32x32 latents, factor 8).
    pixart = PixArtSigmaPipeline.from_pretrained(root / "pixart")
    transformer_config = dict(pixart.transformer.config)
    transformer_config["sample_size"] = 32
    pixart.register_modules(
        transformer=PixArtTransformer2DModel.from_config(transformer_config),
        vae=AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8, 8, 8, 8),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=256,
        ),
    )
    shutil.rmtree(root / "pixart")
    pixart.save_pretrained(root / "pixart")
    print("pixart: rebuilt denoiser and VAE at the 256-pixel configuration")

    wan = WanPipeline.from_pretrained(root / "wan")
    shared = {
        "tokenizer": wan.tokenizer,
        "text_encoder": wan.text_encoder,
        "vae": wan.vae,
        "scheduler": wan.scheduler,
    }
    base = dict(wan.transformer.config)

    def denoiser(**overrides: object) -> WanTransformer3DModel:
        return WanTransformer3DModel.from_config({**base, **overrides})

    # Wan 2.2 A14B runs two denoisers split at a timestep boundary.
    WanPipeline(
        **shared, transformer=denoiser(), transformer_2=denoiser(), boundary_ratio=0.875
    ).save_pretrained(root / "wan22")
    print("wan22: two-denoiser text-to-video composition")

    # Image-to-video denoisers take latents, a frame mask and the encoded
    # source frame (16 + 4 + 16 channels). Wan 2.1 also cross-attends to CLIP
    # image features; Wan 2.2 A14B does not use an image encoder.
    inner = int(base["num_attention_heads"]) * int(base["attention_head_dim"])
    vision = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            patch_size=8,
        )
    )
    processor = CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    WanImageToVideoPipeline(
        **shared,
        image_encoder=vision,
        image_processor=processor,
        transformer=denoiser(in_channels=36, image_dim=16, added_kv_proj_dim=inner),
    ).save_pretrained(root / "wan21-i2v")
    print("wan21-i2v: image-to-video composition with a CLIP image encoder")
    WanImageToVideoPipeline(
        **shared,
        transformer=denoiser(in_channels=36),
        transformer_2=denoiser(in_channels=36),
        boundary_ratio=0.9,
    ).save_pretrained(root / "wan22-i2v")
    print("wan22-i2v: two-denoiser image-to-video composition")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    from huggingface_hub import snapshot_download

    root = Path(sys.argv[1]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name, (repository, revision) in PIPELINES.items():
        snapshot_download(repository, revision=revision, local_dir=root / name)
        print(f"{name}: {repository}@{revision}")
    _compose(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
