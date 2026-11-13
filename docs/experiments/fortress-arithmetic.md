# Fortress Arithmetic and Share-Storage Experiment

**Evidence class:** verified local CPU experiment. These are small measurements, not hardware benchmarks or asymptotic security claims.

Reproduce with:

```bash
.venv/bin/python experiments/fortress_arithmetic.py --iterations 5
.venv/bin/pytest tests/fortress tests/experiments/test_fortress_arithmetic.py -q
```

The experiment uses a deterministic 16×16 integer matrix and five iterations. Exact reconstruction is checked for both full random and ChaCha20 seed-expanded additive shares. Each non-final seeded share uses an independent SHA-256 domain-separated seed. Full shares are labelled information-theoretic only under their independence and non-collusion assumptions; seed-expanded shares are explicitly computational.

## Measured values

| Measurement | Result |
|---|---:|
| Full-share storage | 2048 bytes |
| Seed-expanded storage | 1056 bytes |
| Storage saved by seed expansion | 992 bytes |
| Full-share split | 0.248392 ms |
| Seed-expanded split | 0.475264 ms |
| Prime-field challenge | 0.336992 ms |

The ring is `2^32`; the alternative prime is `2^32 - 5`. The prime measurement covers packed Rust challenge construction only and is not a complete engine-performance comparison. Arbitrary-precision modular products avoid int64 overflow for full random 32-bit shares. Seed expansion trades additional CPU work for less stored share material and must not be described as an information-theoretic one-time pad.

Machine-readable output fields include both storage values, all three timings, both security labels, the matrix shape, and `exact_reconstruction: true`.
