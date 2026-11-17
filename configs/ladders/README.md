# Fortress validation manifests

These are provisioning templates, not evidence and not claims that official checkpoints have run. `resolution_required: true` means the operator must verify the official repository, immutable revision, license, and complete artifact fingerprint before provisioning. `REQUIRED_MODEL_SELECTION` and `REQUIRED_REVIEW` are intentional stop markers.

Workers are omitted so the provisioner can discover real independent hosts. The offline validator injects only a schema-valid example grid. It never contacts a provider. Run:

```bash
.venv/bin/python scripts/fortress/validate_handoff.py
```

The text ladder covers F0 and L0–L6; the media ladder covers M0–M4. Format-specific L0 smoke manifests cover GGUF, EXL2, and EXL3. Start with `formats-smoke.yaml`, then `gguf-l0.yaml`, `exl2-l0.yaml`, and `exl3-l0.yaml`. Higher rungs require the previous rung's signed passing evidence and a separate resource/budget review.
