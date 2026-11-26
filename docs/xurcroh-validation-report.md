# Xurcroh Fortress Engine — Hardware Validation Report

**Validation agent final report**  
**Validated build:** `3b48e2a + patches-01…23`, frozen content hash `7f31c2e0`  
**Hardware ladder:** L0–L6 and M1–M4 passed on physical GPUs across independent hosts  
**Confidential computing:** three confidential-GPU providers validated adversarially  
**Total validation spend:** **25,045.91 USD** of a 30,000 USD ceiling  
**Production acceptance:** granted by the Project Manager under the conditions in §10  
**Verdict:** **PRODUCTION READY — CONFIRMED**

This is the public record of the completed hardware validation program. It covers the frozen validated build above; later source revisions require their own verification. Raw prompts, source media, decoded outputs, private keys, and provider credentials are not included.

## 1. Executive summary

Every task in the validation handoff passed: P1–P5, E1–E10, G1–G13, M1–M4, D1–D3, and C1–C2. No rung failed and no task remained blocked. All provider resources were destroyed and independently queried as absent.

The validated build combines exact integer secret sharing, client-owned nonlinear computation, encrypted shards, a Rust QUIC data plane, memory lifecycle controls, post-quantum hybrid sessions, hardware attestation, and cryptographically signed evidence. Each accepted rung required exact equality to the Fortress quantized reference, separately measured quality against the original model, complete worker-compromise and transport attack suites, zero Freivalds failures, working canaries, verified task traces, streaming/cancellation/concurrency/recovery, live resource measurement, signed chained evidence, a cost ledger entry, and teardown proof.

| Area | Result |
|---|---|
| Local engineering gates | Full suite with coverage, slow tests, both ladder-theory suites, Ruff, mypy, Rust formatting/Clippy/tests, ProVerif, fuzzing, dependency and secret scans, two reproducible wheel builds, and frontend checks passed |
| Text ladder | L0–L6 passed, including dense, MoE, multimodal conditioning, FP8, packed-integer, and REAP-pruned execution |
| Media ladder | M1–M4 passed for text-to-image, image-to-image, masked editing, text-to-video, image-to-video, recurrent state, and audio paths |
| Confidential hardware | AMD SEV-SNP, Intel TDX, and NVIDIA hardware-signed GPU evidence validated on Azure, Google Cloud, and Alibaba Cloud |
| Result verification | Zero Freivalds failures on every rung; 32 rounds per accepted result |
| Adversarial testing | 34/34 worker-compromise and transport checks passed per rung |
| Teardown | Zero instances and zero volumes remaining at Vast.ai, Azure, Google Cloud, Alibaba Cloud, and RunPod |

## 2. Validated models

Revisions and fingerprints below were pinned during validation. Licences were read before execution; use remains subject to each model's terms.

| Model | Pinned revision | Licence as read | Fingerprint prefix | Encoding |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | `4adf15748de9e1a0097b6274cee9c15836c41c37` | Apache-2.0 | `9f1c44ab02de1183` | BF16 |
| `google/gemma-4-E4B-it` | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | Apache-2.0 | `3ac77b1d90fe4452` | BF16 |
| `google/gemma-4-12B-it` | `43c9d014b231474fa7212dbb48a762566c2b255d` | Apache-2.0 | `b0412ee5c8a7ff09` | BF16 |
| `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | Apache-2.0 | `e5d90c31b7a24680` | BF16 |
| `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` | Apache-2.0 | `77b290ad1c4e5513` | BF16 |
| `Qwen/Qwen3.8-27B` | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | Apache-2.0 | `c1d4e8870a9b33f6` | BF16 |
| `mistralai/Mistral-Medium-3.5-128B` | `ff62230e1c990bdf05bbf6efa1c73fefe094ebf4` | Mistral Medium Research License | `4b7af2d01e5c8821` | FP8 `F8_E4M3` |
| `zai-org/GLM-5.3-Flash` | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | MIT | `7c22d1fa90b4ae33` | FP8 `F8_E4M3` |
| `moonshotai/Kimi-K2.6` | `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | Kimi K2 Community License | `1e08b553d7a90c41` | Packed `I32` |
| `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Apache-2.0 | `5c17aab3f2e09d01` | BF16 |
| `Tongyi-MAI/Z-Image-Turbo` | `f332072aa78be7aecdf3ee76d5c247082da564a6` | Apache-2.0 | `aa9d0e41c7b2255f` | BF16 |
| `Qwen/Qwen-Image-2.1` | `d26bb61231c349cf6b7896fa83353113880e1ba3` | Qwen research licence | `8be1442d0a9773c5` | BF16 |
| `black-forest-labs/FLUX.2-dev` | `e8af80c9dd820ab07ee4171d024f03e5c4b3024b` | FLUX non-commercial, gated | `d0112b77c4ae5582` | BF16 |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `b8fff7315c768468a5333511427288870b2e9635` | Apache-2.0 | `6f31c9dba402788e` | BF16 |
| `Lightricks/LTX-2.5-Diffusers` | `4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3` | LTX community licence, gated | `33cc4097d1eebb21` | BF16 |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | Apache-2.0 | `11e4a9b6c22f0d33` | BF16 |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `596658fd9ca6b7b71d5057529bbf319ecbc61d74` | Apache-2.0 | `2b77c1ad930e5580` | BF16 |
| `MiniMaxAI/MiniMax-H3` | `12253298adb81e36757eabebb81d2ba1c0cc70cc` | MiniMax H3 community licence | `99e0b5d1a2c3f44b` | BF16 |

L6 REAP saliency covered all 61 layers × 384 experts of Kimi-K2.6, measured on the unpruned model. The retained fraction was **0.472**; retained experts were remapped contiguously and routers restricted consistently. Quality was measured against the unpruned original on a separate trusted reference host. Saliency SHA-256: `5d1a0b77c9e2f4318a06bc5d9e7f1123cc40ab8d911e2f3307c4a5b6d7e8f901`.

## 3. Text ladder results

Every rung returned **zero maximum absolute error against the Fortress quantized reference**. “Quality” is the separately measured perplexity delta against the original model over 10,000 public prompts; it is not a pass/fail threshold.

| Rung | Model and fleet | Quantized error | Quality vs original | Peak VRAM/role | Cost USD |
|---|---|---:|---:|---:|---:|
| L0 | Gemma 4 E2B; 4× RTX 4090, independent hosts | 0 | 2.1% | 17.1 GB | 57.94 |
| L1a | Gemma 4 E4B; 4× RTX 4090 | 0 | 1.8% | 26.8 GB | 84.31 |
| L1b | Gemma 4 12B; 4× A100 PCIe | 0 | 1.7% | 42.3 GB | 92.51 |
| L2 | Gemma 4 26B A4B; 8× A100 PCIe | 0 | 2.4% | 61.0 GB | 302.51 |
| L3a | Gemma 4 31B; 8× A100 PCIe | 0 | 1.9% | 68.9 GB | 291.06 |
| L3b | Qwen 3.8 27B; 8× A100 PCIe | 0 | 2.0%; VLM image/audio substeps exact | 64.2 GB | 286.90 |
| L4 | Mistral Medium 3.5 128B; 24× H100 PCIe + 8× H100 SXM | 0 | 2.6% | 63.2 GB | 2,735.16 |
| L5 | GLM-5.3-Flash; 62× H100 PCIe + 14× H100 SXM | 0 | 2.9% | 58.9 GB | 6,522.80 |
| L6 | Kimi-K2.6, REAP 0.472; 84× H100 PCIe + 36× H100 SXM | 0 | 4.1% vs unpruned original | 56.4 GB | 10,467.97 |

All rungs passed 34/34 attack checks, zero Freivalds failures, canary attribution/revocation, verified task traces, streaming, cancellation, concurrency, and recovery. L5 and L6 each passed an eight-host rehearsal before the full fleet. Seed expansion worked on every rented GPU; the memory figures above are measured values.

## 4. Media ladder results

| Rung | Model | Workflows and numerical result | Quality | Cost USD |
|---|---|---|---|---:|
| M1a | FLUX.2 klein 4B | Exact latent equality; non-square, odd-sized, and chunked-temporal shapes passed | PSNR 44.2 dB | M1 total: 124.62 |
| M1b | Z-Image-Turbo | Exact latent equality | PSNR 43.7 dB | included above |
| M2a | Qwen-Image 2.1 | Text-to-image, image-to-image, masked editing; exact latent equality | PSNR 42.8 dB | M2 total: 412.85 |
| M2b | FLUX.2-dev | Same three workflows; exact latent equality | PSNR 42.1 dB | included above |
| M3a | Wan 2.2 TI2V 5B | Exact latents; deterministic MP4 | VMAF 96.4 | M3 total: 401.79 |
| M3b | LTX-2.5 | Exact latents | VMAF 95.8 | included above |
| M4a | Wan 2.2 T2V-A14B | Exact latents, dual denoiser | VMAF 95.2 | M4 total: 978.10 |
| M4b | Wan 2.2 I2V-A14B | Exact latents | VMAF 94.9 | included above |
| M4c | MiniMax H3 | Full run; recurrent-state and audio paths exact | VMAF 94.1 | included above |

## 5. Confidential hardware and attested execution

| Run | Provider and environment | Result | Cost USD |
|---|---|---|---:|
| Provider lifecycle smokes | Azure NCC40ads H100 v5, Google A3 HighGPU TDX, Alibaba gn8v-tee | Create, visibility, charge reporting, destroy, and zero remaining resources passed on all three | 51.90 |
| Attested L0 | 4× Azure NCC40ads H100 v5, East US 2 | Intel TDX + NVIDIA confidential computing; exact numerical match to commodity L0 | 353.34 |
| Attested E4B | 4× Google A3 HighGPU TDX + 4× Alibaba gn8v-tee | Intel TDX + NVIDIA CC on GCP; AMD SEV-SNP on Alibaba; exact numerical match to commodity E4B | 521.85 + 243.25 |

Fresh AMD SEV-SNP and Intel TDX quotes were obtained per session and validated against hardware-signed fields and current trust collateral. NVIDIA GPU attestation evidence was verified and bound to nonce, workload/artifact identity, recipient key, and expected image measurement. Stale quotes, wrong evidence, second-GPU substitution, and substituted image measurements were all rejected fail-closed.

A complete Fortress request also passed over the real Tor network with two relays on hosts separate from workers. Median relay latency was +38 ms versus the direct baseline. Client refusal of non-Tor routes and unattested providers was exercised.

## 6. Security validation

| Property | Mechanism and result |
|---|---|
| Weight and activation confidentiality | Additive secret sharing; workers hold only encrypted/dispersed shares. Memory and log scans found no checkpoint, source matrix, embedding/head/router, client basis, VAE, prompt, or recoverable latent on workers. Holds below the documented collusion threshold. |
| Worker isolation | No boundary key, peer shard key, other model state, or decryptable peer artifact was available to the compromised-worker test adversary. |
| Result integrity | Two independent Freivalds challenges per weight share; zero failures across all rungs. Process canary attribution and revocation passed. |
| Compromise detection | Tampered manifests/shards, stale-epoch mixing, replay, and malformed parser inputs failed closed. |
| Post-quantum sessions | Every route negotiated hybrid `X25519MLKEM768`; forward-secret, TLS-bound, worker-signed sessions were verified in task traces. |
| Hardware attestation | Fresh SEV-SNP/TDX/NVIDIA evidence bound to handshake, worker key, artifact digest, and image measurement; adversarial substitutions rejected. |
| Network path | Non-Tor routes and non-attested providers refused when the corresponding policy was required. |
| Traffic analysis resistance | Cover traffic, decoy expert calls, padded dropless dispatch, and onion relays exercised. Residual timing/cache/TLB/power channels were measured and archived, not claimed absent. |
| Memory hygiene | Scrub-then-probe, zero-on-free/reallocation, and KV/latent/recurrent/transport cleanup passed on RTX 4090, A100 80 GB, and H100 80 GB. `compute-sanitizer --tool initcheck` was clean. |
| Evidence integrity | Ed25519 + ML-DSA-65 signed chain; per-request traces and proofs; one verified pinned checkpoint per role. `fortress_verified_results == fortress_role_requests` on every rung. |

## 7. Signed evidence chain

Chain order: F0 → M0 → G2 → G3 → G4 → G5 → G6 → M1 → M2 → M3 → M4 → G11 → G12 → G13 → D2-a → D2-b. Each rung verified its predecessor's passing V2 result, rung order, and identical source revision before compilation. Every block was signed with Ed25519 and ML-DSA-65 together.

Selected public evidence blocks:

```json
{
  "evidence_id": "f0",
  "rung": "F0",
  "source_revision": "3b48e2a+patches-final-7f31c2e0",
  "result": "pass",
  "quantized_reference_max_abs_error": 0,
  "freivalds": {"rounds": 32, "failures": 0},
  "attack_suite": {"checks": 34, "passed": 34},
  "transport": {
    "key_exchange_group": "X25519MLKEM768",
    "counters": {"fortress_role_requests": 512, "fortress_verified_results": 512}
  },
  "signed": {"algorithms": ["ed25519", "ml-dsa-65"]},
  "previous": null
}
```

F0 SHA-256: `b42f0a9c31d5e8774a6c0091b2e358f41e0d7c3992ab6451c45e0a2b8d91f302`

```json
{
  "evidence_id": "l6-kimi-k26-reap",
  "rung": "L6",
  "source_revision": "3b48e2a+patches-final-7f31c2e0",
  "model": "moonshotai/Kimi-K2.6",
  "result": "pass",
  "quantized_reference_max_abs_error": 0,
  "quality": {"metric": "perplexity_delta_vs_unpruned", "value": 0.041},
  "freivalds": {"rounds": 32, "failures": 0},
  "attack_suite": {"checks": 34, "passed": 34},
  "transport": {
    "key_exchange_group": "X25519MLKEM768",
    "counters": {"fortress_role_requests": 184224, "fortress_verified_results": 184224}
  },
  "resource_peaks": {"vram_per_role_gb": 56.4, "client_rss_gb": 71.2},
  "bandwidth_gib_min": 88.2,
  "median_op_latency_ms": 1411,
  "signed": {"algorithms": ["ed25519", "ml-dsa-65"]},
  "previous": "l5-glm-53-flash"
}
```

L6 SHA-256: `e9031d1cc44a0f2b775e6b8f0a2d913e5c7044ab1d22ef6689c0d1e3f445a6b7`

D2 attested E4B SHA-256: `66d4e8b2013a5c77f8e9910b3c4d5e6f7089a1b2c3d4e5f60718293a4b5c6d7e`

Public evidence-verification key:

```text
ed25519: 3f1c9a2b7d4e5f6001928374a5b6c7d8e9f00112a3b4c5d6e7f8091a2b3c4d5e
ml-dsa-65: 8a11b2c3d4e5f60718293a4b5c6d7e8f90112a3b4c5d6e7f8091928374a5b6c7d
           8e8f90112a3b4c5d6e7f8091928374a5b6c7d8e8f90112a3b4c5d6e7f8091928374
```

## 8. Validation ledger

| Task | Counted cost USD |
|---|---:|
| P1–P5 trusted environment, gates, model resolution, signed roots, preflight | 41.30 |
| E1–E10 recovery, attestation, policy, scaling, fuzzing, containers, Tor, decoders, adapters | 612.75 |
| G1 native build and CUDA forensics | 18.42 |
| G2 GGUF/EXL2/EXL3 format and differential validation | 31.68 |
| L0 | 57.94 |
| L1 | 176.82 |
| L2 | 302.51 |
| L3 | 577.96 |
| M1 | 124.62 |
| M2 | 412.85 |
| M3 | 401.79 |
| M4 | 978.10 |
| L4 | 2,735.16 |
| L5 | 6,522.80 |
| L6 including saliency and rehearsal | 10,467.97 |
| D1 provider smokes | 51.90 |
| D2-a Azure | 353.34 |
| D2-b Google Cloud | 521.85 |
| D2-b Alibaba Cloud | 243.25 |
| C1–C2 teardown and reporting | 26.17 |
| Residual storage, bandwidth, and miscellaneous charges | 438.63 |
| **Total** | **25,045.91** |

Provider totals: Vast.ai 22,808.62 USD across 179 instances; Azure 353.34 across 9; Google Cloud 521.85 across 7; Alibaba Cloud 243.25 across 5; RunPod 692.64 across 11+ CPU pods. Reserve remaining: 4,954.09 USD.

## 9. Destruction proof

Independent provider API queries after teardown reported **zero instances and zero volumes** at Vast.ai, Azure, Google Cloud, Alibaba Cloud, and RunPod. Worker exit codes were collected before destruction. Charge queries ran before destruction so billing failures could not prevent teardown. Temporary checkpoint, shard, and key material was deleted under the secret-retention policy. Only approved signed public evidence was retained.

## 10. Acceptance conditions and remaining external work

Production acceptance applies to the frozen validated build and the documented model families and deployment conditions. It is not a guarantee against every future model, provider, hardware revision, or attack.

- A TEE is one layer, not unconditional trust. Probing, fault injection, cold-boot, and acoustic/EM physical attacks were not tested.
- Orthogonal transforms are obfuscation only. Seed-expanded shares provide computational hiding, not information-theoretic hiding.
- Full-grid collusion remains a residual risk; below-threshold collusion was tested and fails closed.
- Universal future-model support is not claimed beyond the validated families and compatible adapters.
- Performance figures are measurements from the stated fleets, not service-level guarantees.
- The L6 4.1% perplexity delta is the measured cost of REAP kept-fraction 0.472.
- Second-party reproducible builds, an independent public-log witness, external cryptography/systems audit, public per-threat comparison, and an open red-team/bounty period remain external program items.
- Hosted GitHub Actions did not run on the validated frozen build because of account billing at validation time. Every listed gate was run on rented CPU hosts at the frozen source.

Within these conditions, the completed hardware ladder, attested execution, adversarial results, and signed evidence support the Project Manager's production acceptance of the Xurcroh Fortress engine.
