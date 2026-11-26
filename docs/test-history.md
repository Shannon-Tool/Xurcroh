# Xurcroh Test and Validation History

This page is the chronological public record of engineering verification and hardware acceptance. Current production evidence is in [`xurcroh-validation-report.md`](xurcroh-validation-report.md). Historical local results below are retained for reproducibility and are not represented as results from the final frozen build.

## Final hardware validation

**Frozen build:** `3b48e2a + patches-01…23`, content hash `7f31c2e0`.

| Stage | Evidence | Result |
|---|---|---|
| Engineering gates | Python suite with coverage, slow tests, text/media ladder-theory suites, Ruff, strict mypy, Rust fmt/Clippy/tests, ProVerif, three ASan fuzz targets, handshake-parser fuzzing (2.3M executions), dependency audits, secret scanning, frontend checks | Passed |
| Reproducibility | Two clean wheel builds and four container images, each built twice | Identical hashes/digests |
| Format validation | Qwen3-0.6B GGUF/EXL2/EXL3; GGUF export differential; official ExLlama CUDA differential | Passed |
| Native CUDA forensics | RTX 4090, A100 80 GB, H100 80 GB; field arithmetic, seed expansion, zero-on-free/reallocation, KV/latent/recurrent/transport cleanup, `compute-sanitizer --tool initcheck` | Passed, zero reported errors |
| Text ladder | L0–L6: Gemma 4 E2B through Kimi-K2.6 REAP on independent physical hosts | All passed; quantized reference max absolute error 0 |
| Media ladder | M1–M4: FLUX.2, Z-Image, Qwen-Image 2.1, Wan 2.2, LTX-2.5, MiniMax H3 | All passed; exact latent equality |
| Confidential hardware | Azure NCC40ads H100 v5, Google A3 HighGPU TDX, Alibaba gn8v-tee | Lifecycle and attested inference passed on all three |
| Adversarial security | Worker compromise, transport attacks, replay, tampering, stale/substituted attestation, wrong keys, second-GPU substitution | 34/34 checks per rung; attacks rejected |
| Result integrity | 32 Freivalds rounds per accepted result | Zero failures across every rung |
| Reliability | Streaming, cancellation, concurrency, packet delay/loss, worker SIGKILL/restart, recovery, multi-model smoke | Passed |
| Resource cleanup | Independent provider queries | Zero instances and volumes at all five providers |
| Spend | Complete provider ledger | 25,045.91 USD of 30,000 USD ceiling |

Production acceptance was granted by the Project Manager for the frozen build under the conditions in the [hardware validation report](xurcroh-validation-report.md#10-acceptance-conditions-and-remaining-external-work).

## Engineering validation milestones

### Local mathematical and end-to-end verification

- **679 Python tests passed, 1 skipped, 36 deselected, 81.40% branch coverage** at `544f03b`, with the 80% coverage threshold enforced.
- Strict mypy, Ruff, 29 Rust tests, frontend interaction tests, production frontend build/content validation, local dependency audits, and secret scanning passed on their recorded revisions.
- F0 and M0 proved the miniature text and synthetic-DiT paths end to end on CPU with measured bandwidth.
- Miniature real-model executions covered dense, MoE, VLM image/audio conditioning, neural image generation, and neural MP4 generation through the public SDK.
- Signed evidence verification detected tampering and enforced predecessor ordering.
- The uncapped video-ladder verifier passed at `639ad69` in 10,548.29 seconds on the local CPU host.
- Clean double wheel builds were hash-identical at `6fed862`.

### Adversarial engineering audit

An eight-scope independent internal audit campaign reviewed attestation, providers, Fortress core, media, transport, memory, deployment, and documentation. It confirmed **4 critical and 14 major findings**; all were fixed with regression tests demonstrated to fail on the unfixed code. Corrections included challenge-bound attestation, pinned trust roots, fail-closed NVIDIA/provider verification, role-artifact attestation policy, ring-overflow enforcement, sealed generator seeds, full-checkpoint DiT fingerprints, authenticated heartbeats, session-close KV scrubbing, and burn-on-read allocator activation. Focused re-verification passed before the full suite above.

### Early physical GPU checks

The initial RTX 4090 program established scoped encrypted-stage execution and native CUDA memory controls on Qwen3-0.6B. It also exposed numerical and multi-host integration defects that were corrected before the final ladder. Earlier two-node SDK requests validated synchronous, streaming, concurrent, cancellation, and recovery behavior but were not counted as numerical passes. Those records remain in the engineering archive as development history, not production evidence.

### Final independent hardware program

The final program used a single frozen source state for the entire ladder. Every rung required:

1. Exact equality to the Fortress quantized reference.
2. Separate quality measurement against the original model.
3. Complete worker-compromise and transport attack suites.
4. Zero Freivalds failures and a working canary.
5. Verified task traces for accepted requests.
6. Streaming, cancellation, concurrency, and recovery.
7. Live resource measurements.
8. Signed chained evidence and a ledger entry.
9. Verified teardown.

L0 established the first four-independent-host physical Fortress result. L2 added MoE routing and decoy expert calls. L3 added VLM image/audio conditioning. L4–L6 scaled to 128B dense, large MoE, and REAP-pruned Kimi-K2.6 execution, with eight-host rehearsals before the largest fleets. M1–M4 established exact latent equality across image and video workflows. D2 repeated L0 and E4B on confidential hardware with fresh hardware-signed attestation and adversarial substitution tests.

## Hosted CI and external assurance

Hosted GitHub Actions did not run on the final frozen build because of account billing at validation time; the same gates were run on rented CPU hosts and their outcomes are recorded above. Second-party reproducibility, an independent public-log witness, external security audit, public per-threat comparison, and an open red-team/bounty period remain external assurance work. They are not represented as completed tests.
