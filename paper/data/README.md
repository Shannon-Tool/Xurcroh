# Validation data provenance

These CSV files transcribe only the public results in `docs/xurcroh-validation-report.md`. They are historical measurements for frozen build `3b48e2a + patches-01…23`, content hash `7f31c2e0`; they are not synthetic observations, current provider quotations, or service-level guarantees.

`text-ladder.csv` leaves parameter count as zero where the public report does not state a model's parameter total. The figure generator must not plot those zero entries as parameter counts. RunPod's `11` instance value encodes the report's `11+` lower bound and is captioned accordingly. In `validation-costs.csv`, “Other program costs” is derived as the published total minus the itemized L0–L6 and M1–M4 rung costs; the four categories reconcile exactly to 25,045.91 USD.

Regenerate plot sources from the repository root with:

```bash
.venv/bin/python paper/scripts/generate_figures.py
```
