# Fortress local ladder evidence

These JSON files are emitted by `tests/fortress/test_f0.py` and `tests/fortress/test_m0.py` when `XURCROH_FORTRESS_EVIDENCE_DIR` is set. They record real local CPU execution over Rust QUIC with mutual TLS, four independently spawned worker processes, exact 32-bit words and modular products in the field `2^32-5`, client-side reduction, and client-side Freivalds verification.

Reproduce with:

```bash
XURCROH_FORTRESS_EVIDENCE_DIR=docs/evidence/fortress \
XURCROH_FORTRESS_REVISION="$(git rev-parse HEAD)" \
XURCROH_FORTRESS_WORKING_TREE_DIRTY=true \
  .venv/bin/pytest tests/fortress/test_f0.py tests/fortress/test_m0.py -q --no-cov
```

The byte counts include encoded tensor-frame and encrypted session-envelope bytes measured by the client. They exclude UDP/IP/TLS handshake overhead and artifact distribution. Timings cover inference calls, not compilation, process startup, or connection establishment; they are not hardware benchmarks or performance targets.

`quantized_reference_max_abs_error` compares distributed execution with the local fixed-point reference. `original_model_max_abs_error` exposes quantization and numerical quality loss against the unquantized miniature model; it is not hidden by the exact-reference assertion. A zero first value does not imply acceptable model quality.

The reports include mixed real/decoy calls, a simulated external observation of an actually transmitted canary share, observer-set revocation, and algebraic collusion/reconstruction results. The canary observation is a local security test, not an independent detection service; attribution names the observer set and does not prove which participant leaked. The attack results do not establish information-theoretic secrecy beyond the stated independent-uniform-share assumptions.

A dirty-tree report is development evidence, not a clean-reproducibility attestation. No physical GPU, independent-host, production-security, or external-audit claim is attached to these files.
