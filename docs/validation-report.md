# Xurcroh Validation Report

## Scope and acceptance status

- Engineering branch: `feat/fortress-implementation`
- Current verification revision: `4c5753a` (rented single CPU host; see “Public log, hosted gateway and anonymity-network carriers”)
- Local implementation: **standard suite and current static checks passed**
- Scoped single-GPU encrypted vLLM stage execution: **passed**
- Independent multi-host numerical inference: **not validated**
- MoE, VLM, image, and video hardware execution: **not validated**
- Production security, performance, and operational acceptance: **not established**
- Current GitHub Actions execution: **blocked before startup by account billing/payment settings**
- `production_accepted`: **false**

The earlier “PRE-GPU IMPLEMENTATION COMPLETE” declaration remains withdrawn. This report distinguishes physical evidence from local tests and does not claim production readiness.

## Local and CI evidence

The latest standard local suite at `544f03b` reports **679 passed, 1 skipped, 36 deselected, 81.40% coverage** (80% threshold enforced by pytest-cov) using `-m 'not hardware and not slow'`; this run postdates the entire adversarial-audit fix wave and supersedes the `5918447` run (621 passed, 80.81%) and the earlier `729ce90` run (431 passed, 81.41%). The one skip is the documented at-least-8-GiB-RAM concurrent datagram relay gate. Slow managed SSH tests and the long video-ladder test have separate actual results; the long video ladder passed in 2,209.98 seconds. This includes the corrected image-grid cover-timeout teardown path. At `dee3180`, an isolated loopback OpenSSH server passes generated-key authentication, host-key pinning, remote execution, SFTP upload, and content readback through the production `AsyncSSHExecutor`. At `0078aa0`, the public SDK completes an actual managed deployment over that SSH transport: four isolated CPU role processes start with distinct ports and per-role mutual-TLS identities, execute one Fortress token through authenticated QUIC, and tear down with no remaining role process or listening UDP port. This is a single loopback host and is not independent physical multi-host validation. At `2bea126`, the same managed path generates an actual PNG through a miniature Diffusers pipeline using four isolated Fortress roles and the public image API, again with no remaining worker process or listening port. At `5809134`, a real miniature Qwen3 MoE checkpoint also completes managed one-token inference over the same SSH path, exercising fused expert routing and protected projections with clean teardown. At `6070dc9`, a managed role is killed with SIGKILL after a successful request; the next request reconnects, restarts the role from its encrypted artifacts, establishes a fresh HPKE session, and completes successfully. No worker process or listening port remains after teardown. The two-model managed workload remains unvalidated because it exceeds this host's memory. Full 12-role managed datagram inference has not completed; the current validated relay evidence is a four-role exact additive-share product through real TCP/UDP relays with clean teardown. Earlier failed-run relays were explicitly terminated. At `06cdda7`, a managed custom VLM checkpoint processes actual image and WAV inputs through client-local projections and completes one-token Fortress inference over four remote roles with clean teardown. This tests the custom boundary path, not official full-tower checkpoint quality. At `80eb3e1`, a miniature `Gemma4ForConditionalGeneration` with the official Gemma4 processor and a real vision tower completes managed image-conditioned inference through the public SDK and four isolated roles. This validates the official architecture path at miniature scale, not the full official checkpoint or physical GPUs. At `b02257c`, the same official-processor model also completes a real WAV-conditioned request through the managed grid using the processor-native `audio` argument, with clean teardown. At `650e359`, a real two-frame, one-step `TextToVideoSDPipeline` completes managed Fortress video generation through the public SDK in 661.02 seconds on the local CPU host, with no remaining workers or ports. An earlier 180-second request budget expired after the denoising step; the test now uses a 900-second budget based on measured latency. This latency is a performance limitation, not a benchmark target. A later `XF3` row-batching experiment passed 14 small transport/media tests but failed the real video workload under transport timeouts and was reverted at `a249a03`; it is not claimed as an optimization. Quinn now uses a 30-second handshake deadline, one-second keepalive, and a bounded 900-second idle ceiling; all 29 Rust tests pass, including both negative certificate cases. At `8b8b144`, seeded CUDA workers use the native ChaCha kernel with canonical field reduction and explicit intermediate-buffer cleanup. A hardware test compares it against the existing CPU expansion; it is skipped on this host and remains hardware-pending. The restored unbatched managed text path passes its real SSH regression. Strict mypy covers 135 source files; Ruff, 29 Rust tests, 32 frontend interaction tests, frontend build/content checks, and local dependency audits pass. The last complete green hosted run was [43409086123](https://github.com/Shannon-Tool/Xurcroh/actions/runs/43409086123), including ProVerif, protocol fuzzing, and the reproducibility job. Newer hosted runs fail before executing any step because GitHub reports failed account payments or a required spending-limit increase.

At `59ff7af`, the public SDK also completed real managed SSH inference with 12 CPU role processes using generated per-operation placement maps and operation-specific encrypted artifacts in 71.82 seconds. This is loopback host evidence with an explicit test-only provider-fraction relaxation, not independent physical infrastructure or a production topology benchmark.

At `e66bef0`, managed Diffusers image generation through the same placed 12-role SSH path passed in 70.77 seconds. The run exposed and corrected a divergence where diffusion previously built an unplaced all-worker grid; text and diffusion now share the same grid-builder, and duplicate role coordinates are rejected. This remains loopback evidence.

Placed managed MoE and official miniature Gemma VLM image inference also passed in 68.37 and 92.71 seconds respectively. At `84a4a53`, three direct four-role QUIC recovery runs passed after connection loss, with coordinated restart and bounded rebinding. Generation-aware recovery and a global worker-shutdown deadline were added through `8f25c3f`. Earlier attempts timed out or failed with `EADDRINUSE`. At `87f457d`, a 30-second native receive deadline allowed hard UDP failure detection and recovery to proceed; the full managed SIGKILL recovery test then passed in 160.03 seconds with clean teardown.

At `1f25f69`, the custom managed multimodal image/audio boundary test also passed in 59.00 seconds after placement and lifecycle changes.

At `1b1fbee`, a real single-worker SSH datagram deployment carried an authenticated QUIC request through the remote and local relays and cleaned up in 6.73 seconds. This validates the deployed relay path itself; a full 12-role relay inference pass is not claimed.

The current real video engine regression passed in **341.43 seconds** with a hard process-memory cap and no remaining workers. Managed multi-host video remains hardware-pending; this local result is not an official-checkpoint or performance claim.

At `23e019e`, another full managed datagram attempt was lost during relay setup after eight SSH forwards reported ready; the host restarted before the remaining forwards or inference completed. It is a failed startup/resource test, not an inference pass or a measured transport stall. The deployment now defaults `startup_concurrency` to 2 for discovery, uploads, role starts, and relay setup, with focused configuration and bounded-executor regressions. The complete 12-role relay inference gate remains unvalidated and must run on a host with measured capacity for all persistent roles and forwarders; it will not be retried on this 6 GiB development host.

The two-model managed test has not completed on this host. At `90a99a6`, its 24 simultaneous role processes exhausted the available 6 GiB RAM, leaving approximately 47 MiB before the test was stopped and cleaned up. A larger validation host or lower-memory worker packaging is required; no multi-model managed pass is claimed. At `51ecfa9`, deployment began aggregating discovered host memory across assigned roles and rejecting insufficient capacity before archive transfer or role launch.

| Check | Actual result |
|---|---|
| Python unit/integration/security tests | **621 passed, 1 skipped, 36 deselected** in 886.71 seconds at `59184478f5bbc30aaf1c96e4cd832fc6fe882186`; slow managed/video tests have separate results |
| Python branch coverage | **80.81%**, above the 80% requirement at `5918447` |
| Ruff format and lint | Passed across 347 Python files at `9043219` |
| Strict mypy, 165 source files | Passed at `9043219` |
| Rust tests | **31 passed** at `9043219` |
| Rust formatting, Clippy, RustSec | Passed at `9043219`; no known advisories |
| Frontend interaction tests | **35 passed** at `9043219` |
| Frontend typecheck, lint, production build, content validation | Passed at `9043219`; 22 sections, equations, links, and Mermaid syntax validated |
| Offline ladder/handoff validation | All 22 command forms validated without provider contact at `9043219`, including signed predecessor arguments and real multi-model smoke |
| ProVerif model and protocol fuzzing | ProVerif confirms activation secrecy, response secrecy, and client/server establishment correspondence at `9043219`; strict result checking is enforced. All three ASan fuzz targets completed without crashes: 2,277,310 frame, 379,534 AEAD, and 30,634 HPKE executions; newer hosted runs are blocked before startup |
| Clean reproducible wheel builds | Matched at `6fed862436625d7871c839db13ce3671d7f270cb` (post-audit delivered SHA; double build, hash-identical): Python `7df71f7eb6104cb4d0b9084f020233a47bf2d6afeca951bceb9f632cee89351f`, native `c9aa759f0153582e11140528ec1d214e8addef8f0a2c4d086987f0f9b82435a2`; the earlier `9043219` evidence is superseded; current hosted rebuild remains billing-blocked |
| `pip-audit` | No known vulnerabilities in audited dependencies at `9043219`; local CPU torch/torchvision and Xurcroh packages remain skipped as detailed below |
| `npm audit` | 0 vulnerabilities at `9043219` |
| Quantized checkpoint reader fuzzing | 100 mutations at `9043219`: 51 accepted, 49 rejected, no unexpected failure |
| Secret scanning | Gitleaks 8.30.1 scanned 434 commits across `--all` at `9043219`; no leaks found; release archive checksum verified. The branch-head scan at `9ab5696` covers 438 commits with no findings |
| Sealed worker image | Static Dockerfile/Kubernetes checks passed at `9043219`; CI builds, exact-role smoke, SBOM, image identity, and Trivy are configured, but no local Docker execution is claimed |

The final Python run used `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on the six-GiB development host with no concurrent build. Earlier resource-contended attempts reached 617 passes and exposed different timing/thread-resource failures while clean Rust builds were running concurrently; each affected test passed in isolation. One intermediate run also exposed an unseeded miniature MoE fixture whose random weights could exceed its 0.02 tolerance; the fixture is now seeded and passed ten isolated runs. The concurrent 12-relay startup validation is explicit about the same host limitation and skips below eight GiB RAM; it remains executable on suitable CI. These reruns are recorded because they are local evidence, but the run above is the final full-suite result.

Fortress request deadlines are now installed as context-local state around text, image, and video execution and checked between additive-grid matrix rows. This closes the observed managed-video case where a single denoising step could continue beyond the public 900-second timeout because the old callback ran only between steps. The check does not claim preemption inside one native kernel call; hardware-bound preemption remains a physical validation concern.

At `1dc0558a35b9dd85d07f50d08193aafe3b01ef2d`, the committed managed-video timeout regression passed in 87.47 seconds, raised `RequestTimeoutError`, and left no worker process or listener. Separate attempts with 900-second and 1,800-second success-path budgets both reached their deadlines cleanly on this six-GiB host. The full success validation is therefore explicit about its resource requirement and skips unless the host has at least eight GiB RAM; it remains pending on a resource-suitable host rather than being misrepresented as a local pass.

The remaining slow managed-SSH subset ran at `9ab5696` with 5 passed and 1 resource-gated skip; its two apparent failures were the deployment fail-closed guard rejecting a dirty working tree (an uncommitted documentation edit present only during that run), not code regressions. On the clean tree at `9b09abdd4660706a5525ea062203837b641a7891`, both tests were rerun and passed in 172.79 seconds with no remaining worker process or listener: `test_managed_role_crash_restarts_and_resumes_inference` and `test_managed_multimodal_boundary_inference_over_real_ssh`.

## Adversarial audit campaign (2026-11-20)

Eight independent read-only audit scopes (confidential-GPU providers, CPU-TEE providers, Fortress arithmetic core, shard compiler and formats, media protection, SDK/provisioning/controller, memory lifecycle and attestation binding, and a test-honesty sweep) reviewed the tree at `c9ef6039447a3609b838d934a56c7fa97b14cdac`. The campaign confirmed 4 CRITICAL and 14 MAJOR defects, each fixed with a regression test that fails on the unfixed code and committed in logically grouped commits `0b13445..d5ffaee`:

- Attestation: IBM DCAP evidence is now bound to a challenger-generated nonce instead of a self-comparison; Intel's SGX root CA SPKI is pinned before DCAP verification (pin value computed from Intel's official certificate distribution point); composite attestation enforces the fresh challenge nonce per component; NRAS requires an explicit `verified: true` and validates the token signature when a verifier key is pinned; the local NVIDIA path fails closed because a worker-supplied signature string cannot establish a hardware signature; the AMD path extracts REPORT_DATA before the temporary directory is closed; OCI measurements must match appraisal reference values. Alibaba attestation now rejects TDX debug mode and non-pass GPU appraisal outcomes and fails closed when appraisal fields are absent. Azure `charges()` queries at resource-group scope with an explicit timeframe, matches full ARM resource IDs, and raises on empty rows instead of silently returning zero; Azure `destroy()` runs network cleanup in a `finally` block and polls deletions to completion.
- Fortress core: `decode_role_artifact` no longer stubs attestation to always-pass — it requires an attestation callback or an explicit `allow_unattested` opt-in that is refused when a policy is present; `worker_cli` exposes the flag and SDK deployments pass it only outside TEE mode. The production grid execute path enforces the unreduced-product overflow bound (inner dims above ~131k for 12-bit activations and 4-bit weights are rejected instead of silently wrapping). JSON role coordinates must be genuine integers.
- Memory and lifecycle: per-request runner caches are zeroed and dropped on session close and stage shutdown; the burn-on-read zero-on-alloc allocator is activated during Fortress CUDA worker startup and fails closed in strict mode; the controller exposes an authenticated heartbeat route so workers do not expire one TTL after registration, and expired circuits are purged from list and count paths.
- Media: the diffusion generator seed is transmitted only inside the sealed tensor packet so the latent basis/permutation derivation is not reconstructable from plaintext parameters; DiT checkpoint fingerprints cover all tensors; safetensors weight-map traversal is rejected; the DiT compiler seed defaults to CSPRNG.

Focused re-verification of every touched area passed at `d5ffaee583c903acbea7bdadaf267a07df5bfd98`: providers and controller 88 passed; security 62 passed; shards, formats, and runtime 149 passed; fortress 184 passed with 1 resource-gated skip; models and integration 81 passed; client, SDK, adapters, and experiments 63 passed; Ruff check and format clean. The full standard suite with coverage enforcement has since been rerun on the delivered SHA `544f03b` (679 passed, 81.40% coverage), and the uncapped video-ladder verifier rerun passed on the delivered code SHA in 10,548.29 seconds; both results are recorded above. Two audit notes are carried forward: Alibaba appraisal field names follow Alibaba's documented EAT profile and must be confirmed against a live token on the first hardware rung, and the Intel root pin must be updated if Intel rotates the SGX root key. Remaining MINOR findings (test-honesty nits such as zero-assertion tests and bare `pytest.raises`, Nitro relay source filtering) are recorded in the audit and queued for a follow-up pass; none creates a verified security or correctness hole.

The uncapped video-ladder verifier rerun was diagnosed after 2.6 hours with py-spy: it was not stalled but spending unbounded time in `measure_timing_classifier` (`python/xurcroh/fortress/attacks.py`), whose leave-one-out loop rebuilt the remaining observation list per sample — O(n²) in the number of timing observations, which the video rung produces in large volume. The classifier is now O(n) with results proven identical to the naive implementation by property-equivalence tests (commit `3f26d35`); 50,000 observations classify in well under a second where the old code needed minutes. The rerun was restarted on the fixed tree; the killed 2.6-hour attempt is recorded as a failed run caused by this defect, not as evidence. The uncapped rerun subsequently **passed at HEAD `639ad69cfe1f043f1a4aac448f1404949f589a42` in 10,548.29 seconds (2:55:48)** on the idle local host — this is a local CPU run, not a hardware or benchmark claim.

The integrated deployment now generates per-operation placements and packages only assigned artifacts, uses operation-specific SDK roles, supports verified external-provider teardown, records live role resources, and checks GPU capacity before launch. Media verification covers configured multi-denoiser and editing/I2V workflows. Text verification covers streaming, cancellation, concurrency, and real multi-model smoke. Original models load only after compiled references are released. Signed predecessor evidence is mandatory for every non-root rung, and operator-saliency REAP pruning consistently filters expert reads and routers. These are local implementation and miniature-model results, not official full-checkpoint or independent-host acceptance.

`pip-audit` explicitly skips the local CPU PyTorch/torchvision wheels and local Xurcroh packages because they are not available from PyPI; those packages are not silently described as audited. Docker/Kubernetes manifests have static tests, but no local Docker image build or cluster application is claimed. The prior wheel comparison preserved Maturin's CycloneDX dependency inventory while normalizing checkout paths, compiler paths, file modes, ZIP encoding, and `RECORD`. Its cross-host SHA-256 values were `57a3e9c309ac80ed575eeedee09ea64d8cf13d2ff6b3f6ca3d3feb3a335cbaa0` for the Python wheel and `113652e94e084a89c0908a3dedc8a09ce843edd29258ce0c90eb85c93232b6c2` for the native wheel. This is automated cross-machine evidence, not second-party attestation. The current clean-build comparison at `9043219ea1e199db08f8e8583a1bcab7bbf8f1ca` produced matching SHA-256 values `76b45d038ccdde714bd555932663ac3b7c072b4f114ae44c36b4c76be776f2c1` (Python) and `c9aa759f0153582e11140528ec1d214e8addef8f0a2c4d086987f0f9b82435a2` (native). No `python/` runtime source changed after that build; later commits affect tests, provider smoke tooling, or documentation only.

Primary commands:

```bash
.venv/bin/pytest --cov --cov-report=term -m 'not hardware and not slow' -q
.venv/bin/ruff format --check python benchmarks tests examples
.venv/bin/ruff check python benchmarks tests examples
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
- Protocol: the current ProVerif model passes all three local queries under its documented authenticated-establishment assumptions; strict summary checking rejects any false or missing result. Bounded frame/AEAD/HPKE fuzzing passed in the recorded runs. Noncanonical unused shape slots are rejected.
- Local and GitHub-hosted wheel builds are byte-identical across hosts. Second-party reproducibility, a live public transparency log, external audits, physical multi-GPU execution, independent attestation, external cryptography review, and production acceptance remain open and are not claimed.

## Quantized formats and current media/handoff evidence

- GGUF v1–v3 ingestion has 64 focused tests covering all 37 current type codes, shards, metadata arrays, embedded tokenizers, and both endian modes. EXL2/GPTQ and EXL3 differential fixtures cover mixed widths, activation ordering, every integer trellis rate, fractional rates, all codebooks, and packed sign vectors. The complete format/preflight suite reports 87 passed tests. These are synthetic/source-layout results, not official full-model or CUDA results.
- GGUF, EXL2, and EXL3 each complete full miniature model compilation and actual chat through `from Xurcroh import Xurcroh`; GGUF also passes synchronous streaming and async-client execution. Fused QKV, stacked experts, metadata-only shape discovery, and incompatible plaintext-engine rejection are covered. Unknown graph semantics fail closed.
- Multi-denoiser media operations use namespaced placement/artifact IDs. The real image ladder returns exact quantized latents; I2V and masked-editing parameter tests pass. The real video regression passes in 341.43 seconds with a memory cap. Full 12-role relay and two-model managed runs remain unresolved; the four-role relayed grid returns an exact product.
- The interactive quantized-format laboratory passes 35 frontend tests plus production build/content validation. All 21 offline ladder manifests validate without provider contact. V2 signed evidence requires source/decoder provenance, workflow, resources, relay mode, and teardown. Official ExLlama CUDA differential tests are executable and currently skip on this non-CUDA host; no official decoder hardware result is claimed.

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
8. Validate automatic per-operation placement on independent physical infrastructure. Managed deployment now generates and applies `PlacementMap` during artifact packaging; local implementation tests do not establish physical diversity or non-collusion.

Orthogonal transformations are computational obfuscation, not homomorphic encryption, cryptographic secret sharing, zero knowledge, or a non-reversibility proof. No production certification is issued by this report.

## Physical harness retry evidence

The first new Fortress CUDA baseline attempt (`60505214`) reached neither bootstrap nor a CUDA test because SSH refused the connection during initialization. The second (`60506413`) encountered host-key verification failures and reached its bounded SSH timeout. Both failures occurred before any hardware test result could be claimed. Provider teardown verified `remaining_instance_ids: []` in both cases; conservative recorded costs were **$0.062153** and **$0.106403** respectively. These failures motivated the bounded transport-readiness and per-worker host-key-pinning regressions in `scripts/fortress/deploy_ladder.py` and `tests/fortress/test_ladder_ssh.py`.

## Corrective field and loader evidence

The research review exposed a ring/prime mismatch and a low-32-bit CUDA/verifier mismatch. The corrective implementation uses canonical arithmetic, transport, verification, and reconstruction over `2^32-5`, with 32-bit field words, fresh OS-generated challenges, HKDF-separated deployment secrets, and first-failure client quarantine. The corrected Fortress suite reports **105 passed, one long video test deselected**. The prime-multiple forgery and low-32 truncation regressions are included. This is local evidence, not physical-GPU acceptance.

An official Qwen3-0.6B client compile with neither worker matrices nor quantized weight values retained completed in **108.79 seconds** with **3,011,548 KiB maximum RSS**. The earlier FP32 metadata-only compile peaked at 3,333,512 KiB; an FP64-boundary attempt was killed at 3,768,088 KiB. These are process-level measurements on the small development host, not thin-client guarantees.

The CUDA loader diagnostic identified the vLLM image's compatibility library as the immediate failure. On RTX 4090 host driver `615.71.09`, the default process mapped `/usr/local/cuda-12.9/compat/libcuda.so.575.57.08` and returned `cuInit=803`. With the host-library override it mapped `/usr/lib/x86_64-linux-gnu/libcuda.so.615.71.09`, returned `cuInit=0`, found one device, and passed a torch CUDA allocation. The native worker now rejects compat-library or driver-mismatch startup, and the forensic script selects the host library explicitly. The diagnostic instance was destroyed with no remaining instances.

### Native field parity and sanitizer

The `10a12c6` physical run replaced 64-bit division with exact pseudo-Mersenne reduction but still failed the observed product regression. The root cause was then isolated to the Python boundary: signed `int32` conversion followed by remainder modulo `2^32-5` shifts values above `2^31-1` by five. Unsigned 32-bit canonicalization must precede field reduction. The largest-product torch oracle was also corrected because signed int64 multiplication overflows; Python arbitrary-precision arithmetic now supplies the exact expected value, with torch retained as a same-device control for non-overflowing products.

At `87311fe`, the physical RTX 4090 hardware suite reported **20 passed, 2 skipped**, including all native field products, irregular matrix shapes, ChaCha expansion, vLLM execution, and memory lifecycle tests. The overall command still failed because the vLLM image did not contain `compute-sanitizer`. Revision `9936ca9` additionally applies canonicalization to the actual CUDA worker response path and adds a CPU regression for that path.

At `a42b552`, the version-matched sanitizer was provisioned and the complete physical gate passed: **20 hardware tests passed, 2 skipped**, followed by **18 tests passed under `compute-sanitizer`** with `ERROR SUMMARY: 0 errors`. Instance `60556773` was destroyed with `remaining_instance_ids: []` and no tracked teardown errors. This establishes the tested native paths only; it is not proof of register/SRAM sanitization, independent multi-host inference, or production acceptance.

### Independent-grid deployment and relay evidence

The synthetic four-host program is implemented by `scripts/fortress/grid_smoke.py` and deliberately produces unsigned infrastructure evidence, not an official L0 model-pass claim. Physical attempts exposed and corrected provider SSH endpoint mapping, normalized TLS worker names, missing client identities, PEM/DER incompatibility, isolated Rust toolchain conflicts, and SSH process detachment. Regressions cover each corrected path.

Direct provider UDP mappings were created successfully, but both local and gateway-originated QUIC connections timed out. An opt-in SSH-forwarded datagram relay (`datagram_tunnel` in the deployment template) was implemented and passed a real local mutual-TLS QUIC round trip. On the physical grid it carried actual transformed model execution into an MLP, then failed when a cover-traffic role connection timed out. A bounded single reconnect/retry now exists for connection/timeout errors. **No independent multi-host numerical or public-SDK pass is claimed.**

All tracked instances from these attempts were destroyed; a direct provider query returned no remaining instances. Subsequent allocation attempts failed with HTTP 400 while the provider account reported a balance of approximately **-0.08145 USD**. No billing action was attempted. Further physical grid, Qwen, and Gemma validation requires available provider capacity and credit.


## Format, ladder-theory, and rented-CPU evidence

This section records work at `9c18a4d24d3f14791a76bff979eeea705877bf78`. All of it ran on one rented CPU host (16 vCPU, 64 GiB, no GPU) with loopback workers. It is local evidence: it does not establish independent-host, GPU, official-checkpoint-quality, or production results, and no ladder rung is claimed as hardware validated.

### What the previous delivered revision got wrong

New regression tests were run against the previously delivered revision `9169e62`: **46 failed, 9 passed**. They pass at `9c18a4d`. The defects they expose had not been observed because quantized formats had only been exercised with a tiny Llama fixture, and ladder verification only with the default L0 manifest and generic UNet pipelines.

| Area | Defect at `9169e62` | Status |
|---|---|---|
| GGUF semantics | One shared tensor-name table for every architecture: Qwen3 query/key norms dropped, Gemma sandwich norms mis-mapped, Gemma 3 norm offset ignored, Llama query/key rows left in llama.cpp order, Qwen2 biases and tied heads unsupported | Replaced by per-architecture family entries; unknown architectures and tensors fail closed |
| GGUF container | Official Gemma 4 E2B export rejected (more than one million tokenizer array entries; per-layer feed-forward list) | Fixed |
| Quantized loading | Projection biases never installed (uninitialised memory); undefined parameters ignored; a model type without a decoder compiled to a checkpoint with no model | Fixed; all fail closed |
| Role worker | Fixed 64-session limit with one session per operation: any model with more than 64 operations on a worker failed during execution | Capacity scales with assigned operations |
| Text ladder verifier | Gemma 4 rungs (L0–L3) crashed on shared-KV state; the Qwen 3.8 rung could not find its grid clients | Fixed |
| Gemma 4 MoE (L2) | Parallel routed experts with a layer-level router were not compiled | Implemented |
| DeepSeek REAP (L6) | Router per-expert correction bias not pruned with the experts | Fixed |
| Media SDK | Public generation failed for every transformer pipeline: packed (Flux, Qwen-Image) and normalised (Wan) latents were decoded as plain VAE latents | Fixed |
| Wan | Missing `accelerate` and `ftfy`; time embedder failed on parameter-less protected layers; absent optional second denoiser treated as a component | Fixed |
| Media manifest | Qwen-Image image-to-image mapped to the edit pipeline of another checkpoint; Wan image-to-video requested from text-to-video checkpoints | Corrected; separate pinned Wan I2V entries added |

### GGUF export differential

`scripts/formats/gguf_export_differential.py` compared one public export per accepted family with its source checkpoint. Aggregated output with exact repositories and revisions is in `docs/evidence/fortress/gguf-export-differential.json`.

| Family | Export | Config | Tensors | Max tensor error | Logit error (scale) |
|---|---|---|---|---|---|
| `llama` | SmolLM2-135M-Instruct F16 | equal | 272 | 3.0e-8 | 4.6e-5 (33.3) |
| `qwen2` | Qwen2.5-0.5B-Instruct F16 | equal | 290 | 3.0e-8 | 3.6e-5 (22.4) |
| `qwen3` | Qwen3-0.6B BF16 | equal | 310 | 0 | 0 (20.7) |
| `gemma3` | Gemma 3 270M F16 | equal | 236 | 3.0e-8 | 3.5e-4 (131.8) |
| `gemma4` | Gemma 4 E2B BF16 | equal | 600 | 0 | 0 (29.6) |
| `qwen3moe` | Qwen3-30B-A3B Q2_K, first source shard | equal | 1,063 names and shapes | not an equality check (2-bit) | not run |

Top-token agreement was 1.0 in every logit comparison. The official Qwen2.5-0.5B FP16 GGUF has weight values that differ from the current source repository, so a community F16 conversion was used for that family. The `qwen3moe` entry does not establish value equality.

### Ladder theory runs

Every manifest entry ran with a miniature checkpoint of its own model family through four loopback role workers and the real `verify_ladder`, including the grid comparison, adversarial checks, original-model comparison, and the public SDK. Text miniatures are constructed in the test; media miniatures are pinned public test pipelines fetched by `scripts/fortress/fetch_tiny_pipelines.py`, with derived two-denoiser and image-to-video compositions.

| Entries | Result at `22dcac6` (clean clone) |
|---|---|
| Text L0–L6, ten entries (Gemma 4 dense and MoE, Qwen 3.8, Qwen dense, Qwen3 MoE, DeepSeek V3 with REAP) | 10 passed, `quantized_reference_max_abs_error == 0` |
| Media M1–M4, seven entries (PixArt-Sigma; Flux and Qwen-Image text-to-image, image-to-image, masked edit; Wan 2.1 and 2.2 text-to-video and image-to-video) | 7 passed, `quantized_reference_max_abs_error == 0` |
| Media M4 MiniMax H3 | Skipped: no public miniature pipeline; never executed |

Miniature weights are random. These runs show that each rung's graph, placement, verification, and SDK path execute; they say nothing about official-checkpoint quality, memory, bandwidth at scale, or timing. Media miniatures use the smallest sizes their pipelines accept rather than the manifest sizes.

### Suite, slow tests, and builds

| Check | Actual result |
|---|---|
| Standard Python suite, clean clone of `9c18a4d` | **726 passed, 0 skipped, 55 deselected, 81.43% coverage** in 515 seconds. At `22dcac6` the same suite reported 723 passed and 3 failed: two from a decoder check that read a missing VAE config (fixed in `4b67680`) and one because the clone had no `.venv` for a helper script |
| Slow tests, clean clone of `22dcac6` | **28 passed, 1 failed, 1 skipped** in 3,404 seconds. Passing for the first time on a host with enough memory: two-model managed inference, complete 12-role managed datagram-tunnel inference, and managed video generation success. The failure is `test_managed_role_crash_restarts_and_resumes_inference`: after the role was killed, reconnection ended in `QUIC connection failed: handshake timed out` and forced termination. It failed the same way on this host at the previous delivered revision `9169e62`, so it is not introduced by this work, but it passed on the development host at `87f457d`; SIGKILL recovery is therefore host-dependent and is not claimed. The skip is MiniMax H3 |
| Ruff lint and format, strict mypy (166 source files) | Passed at `9c18a4d` |
| Offline ladder/handoff validation | All 24 command forms (23 YAML forms plus multi-model smoke) validated at `9c18a4d` |
| Rust formatting, Clippy, tests | Passed at `9c18a4d`; **31 tests** |
| Reproducible wheel build | Double build hash-identical at `9c18a4d`: Python `62fd682a2409e92c77f71ad2eeb4fba457da15de1c0650a59d2be61badb41ac9`, native `c9aa759f0153582e11140528ec1d214e8addef8f0a2c4d086987f0f9b82435a2` (native unchanged from `6fed862`) |

An earlier slow-test attempt reported seven failures; six were `commit all source changes before Fortress provisioning`, caused by copying uncommitted edits into the tree while the tests ran. Those results were discarded and the run was repeated from a clean clone.

Not rerun at `9c18a4d`: frontend tests (no manual content changed), ProVerif, fuzz targets, `pip-audit`, gitleaks, and the container smoke; their last recorded results above apply to earlier revisions. `accelerate` and `ftfy` were added to the models extra and have not been through `pip-audit`. GitHub Actions remains blocked by the account billing state.

### Open findings for hardware validation

- The trusted client holds one session, one QUIC connection, and one cover-traffic loop per operation per role. This has run with roughly one hundred operations per worker on loopback. Large MoE rungs have tens of thousands of operations; file-descriptor, thread, and cover-bandwidth limits at that scale are unmeasured.
- REAP on DeepSeek V3 prunes by global saliency and so changes router group membership; fidelity against the unpruned model is unmeasured.
- MiniMax H3 has not executed in any form.
- Official ExLlama CUDA decoder comparison and full-size quantized execution remain hardware-pending.

### Rented CPU host ledger

RunPod CPU pod `cxcqov99gwm7wk` (cpu3g, 16 vCPU, 64 GiB, US-CA-2) at 0.64 USD/hour, created 2026-11-21T04:11:43Z. Terminated 2026-11-21T07:51:52Z (3 h 40 min, about 2.35 USD of compute at the listed rate; the provider billing query immediately after termination reported 1.73 USD and may lag). An independent query after termination returned HTTP 404 for the pod and listed no network volumes; the only pod remaining on the account predates this work and was not touched. No GPU was rented. The raw test logs on the pod were lost when it was terminated before they were copied; the figures in this section were read from those logs while it was running, and the GGUF differential JSON was copied beforehand.

## Verifiability features and rented-CPU evidence

This section records work at `94f69a51` on one rented CPU host (16 vCPU, 64 GiB, no GPU) with loopback workers, from a clean clone. It is local evidence only; nothing here is hardware, independent-host, or external validation.

### What was added

- Seeded sampling fix: the text sampler rebuilt its generator from the seed for every token. One generator per request now; the effective seed is reported.
- Process hardening in the trusted client and role workers; opt-in prompt armour in the request path.
- Onion relays with any number of hops, configured from worker labels, with a relay command and a worker onion-key option.
- Signed task traces for chat, image, and video with an offline verifier; trace hashes anchored in a Merkle log with inclusion proofs. The Merkle implementation reproduces the RFC 6962 reference root for the standard eight-leaf vector.
- Two independent Freivalds challenges per weight share for remote role results. A regression test forges a result orthogonal to one challenge; it was accepted before the change and is rejected after.
- The negotiated TLS key-exchange group and peer certificate digest are read from the live connection and recorded per route; the relayed SDK test observes `X25519MLKEM768`.

### Results

| Check | Result |
|---|---|
| Standard suite, `-m 'not hardware and not slow'` | 755 passed, 0 failed, 55 deselected, 81.76% coverage |
| Slow tests, `-m 'slow and not hardware'` | 20 passed, 1 failed, 9 skipped, 2,471.80 s |
| Rust workspace tests | 31 passed, 0 failed |
| `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` | Pass |
| Ruff, format check, strict mypy (169 files) | Pass (development host) |

The first full run at `99dfbbb` had **9 failures**: building a trace fingerprinted the checkpoint, which failed for remote engines whose model exists only on the worker. This was a regression introduced by the trace work, found by this run, fixed at `9a1e281` with a regression assertion, and absent from the run above.

The slow failure is `test_managed_role_crash_restarts_and_resumes_inference` (QUIC handshake timed out after the role was killed). It failed the same way on this class of host at `9169e62` and `22dcac6`, before this work. SIGKILL recovery remains host-dependent and is not fixed.

The nine skips are the media-ladder theory tests, which need miniature pipelines that were not fetched onto this host. They last ran at `22dcac6` (7 passed, MiniMax H3 skipped) and were not rerun at this revision.

### Not rerun or not covered at this revision

- Media-ladder theory tests, frontend tests, ProVerif, fuzz targets, `pip-audit`, gitleaks, container smoke, and the reproducible wheel build.
- Clippy with the Python-binding feature enabled reports nine `type_complexity` and argument-count findings in `crates/transport/src/python.rs`. None is in code added here, and the standard lint gate does not enable that feature.
- An earlier focused local run showed one unexplained failure of `test_generation_config_eos_list_stops_real_local_generation` that did not reproduce in three later runs or in the full suite. It mutates a session-scoped fixture and may be order-sensitive.

### Rented CPU host ledger

One CPU pod, created 2026-11-21T17:28:43Z and terminated 18:56:48Z, at 0.64 USD per hour: about 0.94 USD. Termination was confirmed by a 404 on the pod and an empty volume list. Logs were copied off the host before termination. Cumulative spend against the 10 USD test allowance is about 3.29 USD.

## TEE cloud-provider implementation evidence

At `79582191ece6e690bec403ecdb57cdeae71befee`, built-in provider adapters cover Azure NCC40ads H100 v5, Google A3 HighGPU TDX, Alibaba gn8v-tee, OCI E5/E6 SEV-SNP, AWS Nitro Enclaves, and IBM VPC TDX. The first three are confidential-GPU paths; the latter three are explicitly CPU-only. GCP G4 is unsupported. Each adapter implements discovery, create/destroy, remaining-instance checks, and charge reporting. The standard bootstrap now installs the complete `cloud` extra and the compatible IBM/AWS/OCI dependency set.

The initial adapter slice reported **43 provider tests passed** at `7958219`. Subsequent integration regressions bring the current focused provider/lifecycle total to **52 passed**. These tests use mocked provider APIs and generated certificates, not physical cloud infrastructure. They cover confidential configuration flags, scoped ingress, GCP availability filtering and `UNNEST(labels)` billing with bound parameters, Alibaba OIDC policy/runtime validation, OCI Veraison nonce/measurement appraisal, AWS COSE/certificate/PCR/nonce validation, optional Cost Explorer pagination, attested KMS requests, IBM DCAP delegation, and AF_VSOCK framing. Four relay tests execute real local UDP/socketpair forwarding with peer filtering and opaque datagram round trips. Four smoke-harness tests cover create/charge/destroy and recovery. Teardown tests verify built-in provider registration, charge-before-destroy ordering, continued destruction when billing fails, Azure deletion of the per-worker NIC, public IP, vNet, and NSG after the VM disappears, Alibaba deletion of the attached worker security group after instance termination, and Google deletion of the worker-specific firewall rule after VM teardown.

No physical Azure, Google, Alibaba, OCI, AWS, or IBM instance was launched, no cloud charge was incurred, and no physical provider attestation result is claimed. Provider availability, image compatibility, API response shapes, trust collateral, AF_VSOCK kernel support, cost-export contents, and independent destruction remain hardware-validation gates in `docs/gpu-agent-handoff.md`.

## Worker signatures, post-quantum layers and forward-secret sessions

This section records work at `94ba0e9` on one rented CPU host (16 vCPU, 64 GiB, no GPU) with loopback workers. It is local evidence only: no GPU, independent host, official-checkpoint or production result is claimed, and none of this has been externally reviewed. The worker protocol changed in this work, so earlier transport evidence, including `docs/evidence/fortress/transport.json`, describes the previous protocol.

### What was added

- Hybrid signatures (Ed25519 with ML-DSA-65, both required) for task traces, log tree heads, transparency entries, release manifests, ladder evidence, worker registrations and worker checkpoints.
- A signing identity per role worker, a digest of the artifacts each role loads, and signed session checkpoints that the client checks against its own running hash of the session and stores in the trace. One manifest digest per trace binds model source, format, quantization, share count and role artifacts.
- A role session handshake over X25519 with ML-KEM-768 in which client and worker each contribute a secret, signed by the worker identity when pinned, followed by a one-way key ratchet. It is bound to the TLS connection by a TLS exporter on direct routes and is unbound through relays. An expired session is refused instead of being re-created.
- Onion layers sealed with the same hybrid KEM. The X25519-only Rust onion code was removed.
- A ProVerif model of the handshake over an attacker-controlled channel (`protocol/fortress_role_session.pv`). Its three queries hold with client authentication modelled explicitly; a first version without it failed the response-secrecy query. It does not cover relayed routes.
- NIST ACVP vectors for ML-KEM-768 key generation and ML-DSA-65 key generation and signature verification, checked against the installed library. There is no ML-KEM encapsulation vector, because the library does not expose deterministic encapsulation.

### Results

| Check | Result |
|---|---|
| Standard suite, `-m 'not hardware and not slow'`, at `94ba0e9` | 840 passed, 0 failed, 55 deselected, 82.10% coverage, 399.38 s |
| Slow tests, `-m 'slow and not hardware'`, at `4abcf2a` | 20 passed, 1 failed, 9 skipped, 1,147.79 s |
| Rust workspace tests, working tree later committed as `2d93ae8` | 27 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings`; `cargo fmt --check` after formatting | Pass |
| ProVerif, both session models | All queries true (development host, ProVerif 2.05) |
| Ruff, format check, strict mypy (170 files) | Pass (development host) |

The first standard run, at `4abcf2a`, had **10 failures** (829 passed). All ten were tests that still built 32-byte recipient keys, in files that had not been run on the development host. They were corrected at `94ba0e9`, the only change between the two revisions being those tests, one validator placeholder and one documentation sentence, and the suite was rerun in full. The slow suite was not rerun after that change.

The slow failure is `tests/integration/test_managed_ssh.py::test_managed_role_crash_restarts_and_resumes_inference`, the host-dependent crash-recovery test that also failed on this kind of host before this work. It remains open. The nine skips are the media-ladder theory tests, whose miniature pipelines were not fetched on this host, so they were not exercised on the new protocol. The managed SSH tests that passed start real role worker processes with identity keys through the deployment path.

The Rust test count fell from 31 to 27 because four tests of the removed onion code were deleted; one existing test gained the TLS exporter assertions.

### Not done or not rerun

- The older stage-worker and protected-media sessions and shard recipient sealing are still X25519 only. Converting them was planned and was not done: they share one X25519 identity across many modules that Fortress role workers do not use and that could not be regression-tested here.
- Frontend tests and the manual content check, fuzzing, dependency audit, secret scan, container smoke test and the reproducible wheel build were not rerun. The fuzz targets do not cover the new handshake parser; a Python test feeds it malformed input instead.
- Worker checkpoints, the new session and the hybrid onion layers have not run on GPU hardware or across independent hosts.
- Hosted continuous integration is still blocked by account billing, so the ProVerif step added to the workflow has not run there.

## Public log, hosted gateway and anonymity-network carriers

This section records work at `4c5753a` on one rented CPU host (16 vCPU, no GPU) with loopback workers. It is local evidence only: no GPU, independent-host, deployed-service or production result is claimed, and none of it has been externally reviewed.

### What was added

- A public append-only log of digests with inclusion and consistency proofs, signed tree heads and witness countersignatures. The client can publish every trace hash to it and can refuse a model whose manifest digest is not in it. `xurcroh verify` can ask the live log whether it still holds a proof's history and can require named witnesses.
- A hosted gateway that serves the client to API-key holders, assigns each request a task ID, restricts each trace and proof to the requester who made the task, and signs every trace as hosted and operator-asserted.
- Tor (SOCKS5) and I2P (SAM v3) carriers for role-worker traffic, with QUIC mutual TLS and the role session kept end to end, one Tor circuit identity per worker, and the carrier recorded in the trace.
- A fix to random share sampling. Out-of-range draws were rejected, but the remaining draws were converted to signed integers before reduction, so draws of 2^63 and above landed on the wrong residue. Shares described as uniformly random had a bias of about 2^-32. A deterministic regression test failed before the fix.

### Results

| Check | Result |
|---|---|
| Standard suite, `-m 'not hardware and not slow'` | 874 passed, 0 failed, 2 skipped, 55 deselected, 82.48% coverage, 583.93 s |
| Slow tests, `-m 'slow and not hardware'` | 20 passed, 1 failed, 9 skipped, 2,567.59 s |
| Carrier against the real Tor network | Passed |
| Carrier against the real I2P network | Not run: the daemon crashed |
| Ruff, format check, strict mypy (173 files) | Pass (development host) |

The two standard-suite skips are the real-network carrier tests, which run only when Tor or I2P is named in the environment. The slow failure is again `test_managed_role_crash_restarts_and_resumes_inference`, the host-dependent crash-recovery test, still open; the nine skips are the media-ladder theory tests whose pipelines were not fetched. No Rust source changed in this work, so the Rust checks were not rerun.

**Real Tor.** A Tor daemon on the test host published an onion service forwarding to the worker-side datagram relay, and the client reached it through Tor's SOCKS5 port using the test `tests/integration/test_real_anonymity_networks.py` (the copy later committed as `bd2b9e6`). One mutual-TLS QUIC connection negotiated `X25519MLKEM768`; connecting took 4.37 s, and eight 16 KiB echo round trips took between 1.39 s and 3.76 s each. Client and service were on the same host, so this shows the carrier works through real Tor circuits and gives a first idea of latency; it is not a measurement between separate sites, not a full model request, and not a traffic-analysis result. One model layer is one such round trip per token.

**Real I2P.** The packaged `i2pd` 2.49.0 aborted on this host with `malloc(): invalid size`, including after a clean restart and before any client connected, so the SAM bridge never stayed up. All eight attempts failed for that reason. The I2P carrier has therefore run only against an in-process stand-in for the SAM bridge, and nothing is claimed about it on the real network.

### Not done or not tested

- The public log and the gateway were exercised through in-process HTTP test clients, not as deployed TLS services, and the gateway only in front of a local model. The release-script log options have no automated test.
- The project operates no public log and no witness. Without a witness run by a party other than the log operator, the log is only as trustworthy as that operator.
- The gateway is not attested, sees prompts and outputs, and keeps task ownership and traces in memory only.
- A full Fortress request over real Tor was not run; the SDK-over-carrier test uses a stand-in proxy.
- Still open from the plan: decoy calls, verified placement labels, an in-browser verification page, checking the client's own build against the log, and post-quantum conversion of the older stage-worker, protected-media and shard-sealing paths.
- Frontend tests, fuzzing, dependency audit, secret scan, container smoke test and the reproducible wheel build were not rerun. Hosted continuous integration is still blocked by account billing.

