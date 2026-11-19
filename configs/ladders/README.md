# Fortress validation manifests

These are provisioning templates, not evidence and not claims that official checkpoints have run. `resolution_required: true` means the operator must verify the official repository, immutable revision, license, and complete artifact fingerprint before provisioning. `REQUIRED_MODEL_SELECTION` and `REQUIRED_REVIEW` are intentional stop markers.

Workers are omitted so the provisioner can discover real independent hosts. The offline validator injects only a schema-valid example grid. It never contacts a provider. Hardware deployments require enough workers for the placement policy, not merely one N×N grid. Non-Vast providers use `xurcroh.providers` entry-point adapters with destroy, charge, and remaining-instance verification; external worker metadata is accepted without a Vast record only when complete. Set `startup_concurrency` explicitly only after measuring gateway capacity; the default is 2. Run:

```bash
.venv/bin/python scripts/fortress/validate_handoff.py
```

The text ladder covers F0 and L0–L6; the media ladder covers M0–M4. Format-specific L0 smoke manifests cover GGUF, EXL2, and EXL3. Start with `formats-smoke.yaml`, then `gguf-l0.yaml`, `exl2-l0.yaml`, and `exl3-l0.yaml`. Higher rungs require the previous rung's signed passing evidence at the same source revision and a separate resource/budget review. L4 uses Qwen1.5-110B-Chat (Tongyi Qianwen). L6 requires complete per-layer REAP saliency, not the schema-only example, and uses the DeepSeek model license. Flux/Qwen inpainting and Wan image-to-video have explicit Diffusers pipeline mappings in `media-ladder.yaml`.
