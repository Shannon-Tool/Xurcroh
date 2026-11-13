# Xurcroh Validation Report

## Scope and acceptance status

- Engineering branch: `feat/fortress-implementation`
- Validated implementation revision: `ee3cc2a8aba05f5a720866394e3e8deb03831bee`
- Local implementation: **passed**
- Scoped single-GPU encrypted vLLM stage execution: **passed**
- Independent multi-host numerical inference: **not validated**
- MoE, VLM, image, and video hardware execution: **not validated**
- Production security, performance, and operational acceptance: **not established**
- Current GitHub Actions execution: **blocked before startup by account billing/payment settings**
- `production_accepted`: **false**

The earlier “PRE-GPU IMPLEMENTATION COMPLETE” declaration remains withdrawn. This report distinguishes physical evidence from local tests and does not claim production readiness.

## Local and CI evidence

The latest local suite reports **346 passed, 15 deselected, 82.34% coverage**, excluding only the long video-ladder test that passed separately in 2,209.98 seconds. Strict mypy covers 122 source files; Ruff, 29 Rust tests, 32 frontend interaction tests, frontend build/content checks, and local dependency audits pass. The last complete green hosted run was [43409086123](https://github.com/Shannon-Tool/Xurcroh/actions/runs/43409086123), including ProVerif, protocol fuzzing, and the reproducibility job. Newer hosted runs fail before executing any step because GitHub reports failed account payments or a required spending-limit increase.

| Check | Actual result |
|---|---|
| Python unit/integration/security tests | **346 passed, 15 deselected**; full video ladder passed separately |
| Python branch coverage | **82.34%**, above the 80% requirement |
| Ruff format and lint | Passed |
| Strict mypy, 122 source files | Passed |
| Rust tests | **29 passed** |
| Rust formatting, Clippy, RustSec | Passed; no known advisories |
| Frontend interaction tests | **32 passed** |
| Frontend typecheck, lint, production build, content validation | Passed; 22 sections, equations, links, and Mermaid syntax validated |
| ProVerif model and protocol fuzzing | Passed in the hosted run above; newer runs are blocked before startup |
| Clean reproducible wheel builds | Matched at `a2f5847887cb9954a037d58575d30158e1d878dd`; current hosted rebuild is billing-blocked |
| `pip-audit` | No known vulnerabilities in audited dependencies |
| `npm audit` | 0 vulnerabilities |

`pip-audit` explicitly skips the local CPU PyTorch/torchvision wheels and local Xurcroh packages because they are not available from PyPI; those packages are not silently described as audited. Docker/Kubernetes manifests have static tests, but no local Docker image build or cluster application is claimed. The prior wheel comparison preserved Maturin's CycloneDX dependency inventory while normalizing checkout paths, compiler paths, file modes, ZIP encoding, and `RECORD`. Its cross-host SHA-256 values were `57a3e9c309ac80ed575eeedee09ea64d8cf13d2ff6b3f6ca3d3feb3a335cbaa0` for the Python wheel and `113652e94e084a89c0908a3dedc8a09ce843edd29258ce0c90eb85c93232b6c2` for the native wheel. This is automated cross-machine evidence, not second-party attestation.

Primary commands:

```bash
.venv/bin/pytest --cov --cov-report=term -m 'not hardware' -q
.venv/bin/ruff format --check python benchmarks tests
.venv/bin/ruff check python benchmarks tests
.venv/bin/mypy
PATH=/root/.cargo/bin:$PATH cargo fmt --all -- --check
PATH=/root/.cargo/bin:$PATH cargo clippy --workspace --all-targets -- -D warnings
PATH=/root/.cargo/bin:$PATH cargo test --workspace
PATH=/root/.cargo/bin:$PATH cargo audit
cd apps/web-manual && npm run check && npm audit
./scripts/reproducible-build.sh
.venv/bin/pip-audit
```

## Fortress local execution evidence

- F0: four independent local worker processes, exact fixed-point prefill/decode/sampling, 168 inference role calls, 62,496 bytes sent and 48,768 bytes received, zero quantized-reference error, and all worker exit codes zero.
- M0: four independent local worker processes, exact two-step synthetic DiT execution, 328 inference role calls, 117,592 bytes sent and 89,728 bytes received, zero quantized-reference error, and all worker exit codes zero.
- Public SDK: dense/MoE text grids pass locally and over four real mutual-TLS role workers; real `DDPMPipeline` image generation and `TextToVideoSDPipeline` video generation pass, including public `images.generate` and `videos.generate` calls.
- VLM boundary: local image/audio projections feed the Fortress decoder without entering worker operations.
- Protocol: ProVerif and bounded frame/AEAD/HPKE fuzzing pass in CI. Noncanonical unused shape slots are rejected.
- Local and GitHub-hosted wheel builds are byte-identical across hosts. Second-party reproducibility, a live public transparency log, external audits, physical multi-GPU execution, independent attestation, external cryptography review, and production acceptance remain open and are not claimed.

## Physical hardware and checkpoint

Final numerical run:

- Provider instance: Vast.ai `60267949`, machine `140968`
- GPU: one **NVIDIA GeForce RTX 4090**, 24,564 MiB reported by `nvidia-smi`, compute capability 8.9
- Software: vLLM 0.15.0, PyTorch 2.9.1+cu128, CUDA 12.8
- Checkpoint: `Qwen/Qwen3-0.6B` at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Checkpoint fingerprint: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`
- Architecture: 28 decoder layers, hidden size 1024, 16 query heads, 8 KV heads, head dimension 128, QK normalization, half-split RoPE with base 1,000,000
- Execution: ten independently encrypted FP32 stage shards loaded through the Xurcroh vLLM model loader and executed **sequentially on the single GPU**

This was a real official-checkpoint GPU-kernel execution. It was not a plaintext model server, an FP64 production fallback, a ten-host run, or an independent distributed-inference result.

## Numerical method and results

For every stage, the optimized FP32 vLLM result was compared with the FP64 transformed reference using the same encrypted shard. The test exercised a nine-token prefill and a one-token cached decode, client basis transitions, and separate vLLM `hidden_states`/`residual` handoffs. The comparison used `atol=0.05` and `rtol=0.05`; the exact maximum absolute errors are below.

| Stage | Decoder layers | Prefill max abs error | Decode max abs error | Result |
|---:|---|---:|---:|---|
| 0 | 0–2 | 0.02358993045720581 | 0.0007348155005707513 | Passed |
| 1 | 3–5 | 0.0016842429184300012 | 0.0017092329201880974 | Passed |
| 2 | 6–8 | 0.002158098456818358 | 0.0014943944610263105 | Passed |
| 3 | 9–11 | 0.0032755150834524738 | 0.00243413761839939 | Passed |
| 4 | 12–14 | 0.0023941324874161563 | 0.001689940219576691 | Passed |
| 5 | 15–17 | 0.004259051769709199 | 0.004608339565709496 | Passed |
| 6 | 18–20 | 0.007272388612726388 | 0.008047943590678841 | Passed |
| 7 | 21–23 | 0.013932506148063517 | 0.012887488989617779 | Passed |
| 8 | 24–25 | 0.011783098463133967 | 0.011201372677845711 | Passed |
| 9 | 26–27 | 0.015284089845346571 | 0.013424198859590675 | Passed |

The largest optimized-vLLM cached-decode versus vLLM full-forward difference was `0.0012044906616210938`. The largest FP64 cached-decode versus FP64 full-forward difference was `2.842170943040401e-14`.

These values establish numerical agreement with the transformed FP64 reference for this input and precision. They are not a model-wide error bound, BF16 quality assessment, original-Hugging-Face-logit comparison for the full checkpoint, or proof of output quality for other prompts and models.

## Defects found and corrected

GPU validation exposed an inter-stage residual bug. Xurcroh previously fused `hidden_states + residual` and then sent both tensors to the next vLLM stage. vLLM's `Qwen3DecoderLayer.forward` fuses them again at the next layer, double-counting the residual. Revision `9f9249b` now preserves raw hidden states between stages and fuses the residual only before client sampling. Two regression tests failed before the fix; the complete vLLM adapter suite now has 19 passing tests.

The validation harness also initially supplied the previous layer's FP64 cache to decode instead of the current layer's prefill cache. That harness error was isolated by comparing cached and full reference computations; production `DenseStageRunner` already used the correct layer index. The corrected FP64 cached/full differences are at floating-point roundoff scale above.

## Unsuccessful and incomplete physical checks

- An earlier two-node SDK run completed synchronous, streaming, concurrent, cancellation, and post-cancellation request behavior on two RTX 4090 workers. Its numerical comparison **failed** with stage-0 maximum absolute error `3.987569914530866`. It is behavioral evidence only, not successful distributed numerical validation.
- A later single-worker numerical attempt **failed** with maximum absolute error `28.58909221453399`. It was superseded only after the decode and residual defects were corrected and the ten-stage run passed.
- Earlier numerical failures and provider/SSH failures are not counted as successful hardware tests.
- No current two-node or ten-node independent-host numerical pass is claimed.
- No physical SGLang MoE, xDiT, VLM, image-diffusion, or video-generation result is claimed.
- No accepted end-to-end latency or throughput benchmark is claimed from the numerically failed SDK runs. The ten-stage sequential wall time includes SSH, transfers, repeated engine startup, and local development work, so it is not a meaningful inference benchmark.

## Cleanup evidence

After the final stage exited, `nvidia-smi` reported **1 MiB / 24,564 MiB** GPU memory used and no Python worker processes remained. This is process/GPU-release evidence for this run, not proof of register/SRAM sanitization or completion of the allocator forensic suite.

## Vast.ai spending ledger

Provider-reported charges currently total **$9.327**. The budget guard's larger of reported charges and elapsed-time estimates totals **$10.077028235170287**, below the **$25 target** and **$30 hard limit**. Final invoices can arrive later, so `charges_finalized` remains **false**.

| Instance | GPU | Reported USD | Conservative USD |
|---:|---|---:|---:|
| 60162500 | RTX 4090 | 0.657 | 0.657000 |
| 60171346 | RTX 4090 | 0.000 | 0.000320 |
| 60171347 | RTX 4090 | 0.000 | 0.000426 |
| 60171632 | RTX 4090 | 0.000 | 0.000028 |
| 60171974 | RTX 4090 | 0.001 | 0.004395 |
| 60171975 | RTX 4090 | 0.001 | 0.004684 |
| 60172183 | RTX 4090 | 0.682 | 0.682000 |
| 60172193 | RTX 4090 | 0.836 | 0.836000 |
| 60199827 | RTX 4090 | 0.019 | 0.132529 |
| 60199828 | RTX 4090 | 1.374 | 1.374000 |
| 60202022 | RTX 4090 | 1.123 | 1.123000 |
| 60218856 | RTX 4090 | 0.570 | 0.613895 |
| 60218857 | RTX 4090 | 3.170 | 3.170000 |
| 60228615 | RTX 4090 | 0.005 | 0.124168 |
| 60230922 | RTX 4090 | 0.008 | 0.234426 |
| 60235070 | RTX 4090 | 0.016 | 0.255156 |
| 60267949 | RTX 4090 | 0.865 | 0.865000 |
| **Total** | | **9.327** | **10.077028** |

## Destruction proof

The teardown command refreshed provider charges, destroyed every tracked instance, stopped the watchdog, and then listed account instances again. A subsequent status/charge refresh also completed.

- Tracked instance IDs: all 17 IDs in the ledger above
- `all_destroyed`: **true**
- `remaining_instance_ids`: **`[]`**
- `errors`: **`[]`**
- `unrelated_instances_untouched`: **true**
- Watchdog PID after teardown: **none**
- Verification timestamp recorded by the provider lifecycle report: `1794441674.6999094`

No tracked billable test instance remains. This does not assert that unrelated account resources were deleted or that the provider's final invoice has settled.

## Remaining production blockers

1. Repeat corrected numerical inference through the public SDK on at least two genuinely independent GPU workers, then test larger independent-host pipelines.
2. Measure end-to-end prefill/inter-token latency, throughput, transport overhead, memory, cancellation, and recovery under distributed load. The 40–50 ms/token target remains unverified.
3. Validate BF16/FP16 output quality and compare full-checkpoint outputs with the original model across representative prompts.
4. Run physical SGLang MoE, xDiT, VLM, image, and video workloads with official checkpoints and license review.
5. Complete CUDA zero-on-free, KV-canary, stream-ordering, shared-memory/local-memory, and `compute-sanitizer` forensics. Process exit and low post-run memory are not complete memory-sanitization proof.
6. Validate physical AMD/Intel/NVIDIA attestation, trust roots, freshness, artifact binding, and failure handling.
7. Complete WAN/NAT failure injection, multi-user isolation, worker recovery, and side-channel evaluation.

Orthogonal transformations are computational obfuscation, not homomorphic encryption, cryptographic secret sharing, zero knowledge, or a non-reversibility proof. No production certification is issued by this report.

## Physical harness retry evidence

The first new Fortress CUDA baseline attempt (`60505214`) reached neither bootstrap nor a CUDA test because SSH refused the connection during initialization. The second (`60506413`) encountered host-key verification failures and reached its bounded SSH timeout. Both failures occurred before any hardware test result could be claimed. Provider teardown verified `remaining_instance_ids: []` in both cases; conservative recorded costs were **$0.062153** and **$0.106403** respectively. These failures motivated the bounded transport-readiness and per-worker host-key-pinning regressions in `scripts/fortress/deploy_ladder.py` and `tests/fortress/test_ladder_ssh.py`.
