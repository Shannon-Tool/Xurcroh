# Fortress local ladder evidence

These JSON files are emitted by `tests/fortress/test_f0.py` and `tests/fortress/test_m0.py` when `XURCROH_FORTRESS_EVIDENCE_DIR` is set. They record real local CPU execution over Rust QUIC with mutual TLS, four independently spawned worker processes, exact signed 128-bit role products, client-side reduction, and client-side Freivalds verification.

Reproduce with:

```bash
XURCROH_FORTRESS_EVIDENCE_DIR=docs/evidence/fortress \
XURCROH_FORTRESS_REVISION="$(git rev-parse HEAD)" \
XURCROH_FORTRESS_WORKING_TREE_DIRTY=true \
  .venv/bin/pytest tests/fortress/test_f0.py tests/fortress/test_m0.py -q --no-cov
```

The byte counts include encoded tensor-frame and encrypted session-envelope bytes measured by the client. They exclude UDP/IP/TLS handshake overhead and artifact distribution. Timings cover inference calls, not compilation, process startup, or connection establishment; they are not hardware benchmarks or performance targets.

`quantized_reference_max_abs_error` compares distributed execution with the local fixed-point reference. `original_model_max_abs_error` exposes quantization and numerical quality loss against the unquantized miniature model; it is not hidden by the exact-reference assertion. A zero first value does not imply acceptable model quality.

The reports do not by themselves complete F0 or M0. Attack-suite and attributed-canary evidence, reproducibility on a clean revision, and the remaining roadmap acceptance gates are still required. No physical GPU, independent-host, production-security, or external-audit claim is attached to these files.
