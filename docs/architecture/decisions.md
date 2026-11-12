# Architecture and Technical Decisions

## ADR-001: Security claims and threat model

Xurcroh provides computational basis obfuscation, authenticated transport confidentiality, partitioned model custody, and optional hardware-backed isolation. These are distinct properties. Orthogonal or contragredient transformations do not provide semantic security against a worker that observes weights and activations, do not prevent collusion, and preserve structural invariants unless the specific transformation removes them. ChaCha20-based masking uses a pseudorandom generator and is not an information-theoretic one-time pad.

The controller receives only opaque application ciphertext and routing metadata needed for the selected circuit mode. Global passive traffic analysis, endpoint identity, timing, token counts, availability attacks, malicious worker computation, physical TEE attacks, and colluding workers remain separate threats. Constant-load MoE dispatch addresses network volume leakage, not all timing, cache, TLB, or power side channels.

## ADR-002: Zero-shot conversion versus adaptation

RMSNorm coordinate gains are absorbed into immediately adjacent linear projections before conjugation. This is exact when the consumer set is known and no other operation consumes the normalized output. SwiGLU and GeGLU permit intermediate permutations and the documented contragredient scaling of the non-gated branch. Q/K transformations must commute with every positional rotation pair and share transformations across each GQA group. Arbitrary dense residual rotations are not valid for coordinate-wise AdaLN; zero-shot diffusion uses signed permutations, while dense residual bases require separately trained block-scalar modulation.

## ADR-003: Framework integration

The inspected vLLM `BaseModelLoader` exposes `create_model`, `load_weights`, and `load_model`; Xurcroh uses the inspected interface rather than copying the specification's older override. Optional integrations import framework modules only at runtime and fail with an explicit compatibility error if the pinned API is absent. CPU reference execution is real model execution for validation, not a substitute claim that vLLM, SGLang, or xDiT GPU integration has been hardware-verified.

Verified references on 2026-11-10 include:

- vLLM BaseModelLoader: https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/base_loader/
- Gemma 4 Unified: https://huggingface.co/docs/transformers/model_doc/gemma4_unified
- Qwen 3.8 official repository: https://github.com/QwenLM/Qwen3.8
- SGLang Diffusion CLI: https://docs.sglang.ai/docs/sglang-diffusion/api/cli

## ADR-004: Memory scrubbing scope

Zero-on-free synchronizes the wipe on the allocation's stream before `cudaFree`, preventing reuse through that allocator before the wipe completes. KV pools require separate physical-slot wipes because returning a logical block does not call the allocator. An independent epilogue kernel cannot prove that every hardware register, all SRAM, or another device's local memory was cleared. The audit probe measures the tested allocation/shared-memory region only and its report must state that scope.

## ADR-005: Integrity verification

Freivalds' probability bound applies to exact arithmetic over the selected finite field. BF16/FP16/FP8 kernels round and may reorder operations, so the same bound is not asserted for them. Floating-point verification uses documented numerical tolerances and empirical differential tests, while finite-field checks are available for quantized/integer projections.

## ADR-006: Repository and dependency layout

The implementation uses one importable `python/xurcroh` package instead of duplicating modules across suggested directory names. Rust is split into protocol, crypto, and transport crates. The web manual is isolated under `apps/web-manual`; shared machine-readable schemas and evidence reports avoid manually duplicating protocol and model constants. The stable public package is `python/Xurcroh`, exporting `Xurcroh` and `AsyncXurcroh` over those private interfaces.

## ADR-007: Checkpoint-driven model architecture detection

Live checkpoint metadata on 2026-11-10 contradicts two blueprint assumptions. `Qwen/Qwen3.8-27B` declares `model_type=qwen3_5` and `Qwen3_5ForConditionalGeneration`, 24 query heads with head dimension 256, 16 linear key heads and 48 linear value heads, and an empty `deepstack_visual_indexes` list. `google/gemma-4-12B-it` declares 16 query heads, head dimension 256, global head dimension 512, and tied embeddings rather than the blueprint's 24 heads at dimension 160. Adapters therefore resolve behavior from a pinned checkpoint's `config.json` and revision, validate required fields explicitly, and never silently substitute the blueprint constants. Sources:

- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
- https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json

## ADR-008: Public SDK and production boundaries

The required `from Xurcroh import Xurcroh` API is a lifecycle-owning facade over actual execution adapters. It does not contain model responses or simulation calls. Typed configuration accepts existing endpoints and explicitly authorized SSH bootstrap settings while redacting secrets. Methods validate model capabilities and fail before execution when no real adapter exists. Synchronous, asynchronous, and streaming APIs share one request/session state machine rather than maintaining divergent implementations.

Python owns the SDK, ML adapters, and control plane; Rust owns protocol, AEAD, and QUIC; C++/CUDA owns native memory operations; TypeScript is confined to the independent manual. Cross-language boundaries specify buffer ownership, cancellation, errors, and protocol versions. No additional language is introduced without a measured benefit.

## ADR-009: Dense stage protocol and checkpoint RoPE layout

Compiled dense manifests can be executed by actual remote stage workers through `DenseStageEngine`. Each link uses the unwrapped per-stage key with directional HKDF context; the manifest master key is not reused as the tensor-session key. Workers validate encrypted shard identity, checkpoint fingerprint, tensor descriptors, and replay state. Clients retain tokenization, embedding, final normalization, LM-head sampling, and decoded text. The reference network path uses FP64 tensors over development HTTP in local tests or TLS in deployment. Rust QUIC remains the high-throughput transport implementation, but physical GPU-stage QUIC timing and framework-kernel integration are hardware-pending claims.

Strict mode no longer equates HTTPS with zero-trust model custody: full-model remote checkpoints require explicit development policy, while strict remote execution uses compiled stages. Llama-class and Qwen3 checkpoints use half-split RoPE; adjacent-pair and half-split layouts are explicit implementation parameters and the selected layout is authenticated in stage metadata. Qwen per-head Q/K RMSNorm rules out signed rotations because coordinate-wise products do not commute with sign flips. The compiler therefore uses unsigned RoPE-pair permutations for these models and serializes the transformed Q/K norm gains. Boundary bases are always stored at `compute_dtype`, independently of reduced-precision weight storage, and client boundary operations cast to that basis precision before matmuls. Stage manifests record `storage_dtype`; optimized vLLM execution selects FP32 kernels for FP64/FP32 shards, BF16 for BF16, and FP16 for FP16. Both legacy `rope_theta` and nested `rope_parameters.rope_theta` checkpoint metadata are supported. Direct Hugging Face comparison establishes numerical equivalence within `atol=3e-8, rtol=1e-6` for the miniature FP64 Llama fixture and `atol=1e-7, rtol=1e-7` for the miniature FP64 Qwen fixture, not universal bitwise or reduced-precision equivalence.


## ADR-010: Recipient isolation and client-side stage transitions

Manifest v2 seals each stage data key to a distinct X25519 recipient and encrypts the client boundary with an independent client-only key. Request transport uses ephemeral X25519 sessions instead of shard keys. Worker artifacts contain only their assigned transformed weights and cannot authenticate the client boundary or another recipient's envelope using their own identity. These are credential-isolation properties, not a claim that transformed weights resist all alignment or reconstruction attacks.

Worker shards do not contain `stage.input_basis`, `stage.output_basis`, or `stage.transition`. An attempted optimization that folded the next basis only into the last FFN projection was rejected by numerical tests: the layer's residual connection remains in the previous basis, so this transformation is not equivalent. The trusted client therefore applies `hidden @ bases[i] @ bases[i+1].T` between worker calls. This preserves the residual path and exact reference logits without exposing transition matrices to workers, at the cost of client-side matrix multiplication between stages.


## ADR-011: Compiled equivariant media and modulation coefficients

Strict remote image/video execution uses compiled DiT weights and does not send `basis`, `permutation`, or inverse transformations in worker requests. Input and output projections carry the latent basis; joint attention uses shared query/key transforms and value/output transforms; spatial permutation is undone only on the client. Generic UNet execution that recovers latents before every model call is development-only `transport_only` behavior, not a zero-trust mode.

Signed-permutation AdaLN requires coefficient-specific handling. Multiplicative `gamma` and `alpha` use unsigned permutations, while additive `beta` uses the signed permutation. Applying signs to every coefficient is not equivalent because products of two transformed tensors cancel the signs. The compiler and tiny double-stream MMDiT tests verify joint attention, AdaLN, FFNs, input/output projections, spatial permutation, and flow steps against the original model. These tests establish functional equivalence and request-secret absence, not immunity to weight alignment, position/metadata leakage, or model-reconstruction attacks.


## ADR-012: Native-dtype shards and tensor transport

Dense compilation can perform transformations in FP64 and emit BF16/FP16 shards and client boundaries through `shard_dtype`; FP64 remains the reference configuration. Stage descriptors support FP64, FP32, FP16, and BF16 with explicit shape, element-size, and payload-length validation on both HTTP and QUIC. `CudaStage` uses pinned host buffers and stream synchronization for device-to-host staging and cleanup. CPU tests verify native-dtype serialization, shard emission, and public-SDK inference. CUDA execution and physical GPU memory behavior remain hardware-validation claims, not results inferred from CPU tests.


## ADR-013: vLLM v1 stage execution contract

Optimized dense stages use vLLM 0.15.0 `GPUModelRunner` with recipient-sealed shard loading and `enable_prompt_embeds`. Each request creates real scheduler metadata for prefill/decode and returns `IntermediateTensors` containing both `hidden_states` and `residual` to the next worker. The compatibility manifest records source digests and required cache/scheduler settings. `engine="vllm"` and strict `auto` require the `vllm-stage` capability; `engine="reference"` is development-only. Plaintext model-server startup is not accepted as protected execution. CPU tests validate scheduler, recipient, and bundle contracts. Actual CUDA kernel execution and numerical quality require the explicit hardware test and must not be inferred from these contracts.


## ADR-014: SGLang private MoE stage contract

Optimized MoE/long-context stages use SGLang 0.5.23 `TpModelWorker` with `LoadFormat.PRIVATE`, a `DefaultModelLoader` subclass that reads only the recipient-sealed stage. `ForwardBatch` carries transformed `input_embeds`, and `PPProxyTensors` preserves `hidden_states` plus `residual` for the next worker. RadixCache is disabled; KV/request cleanup wipes token slots before returning them to the free pool. CPU tests validate private loading, forward/proxy contracts, version rejection, and cleanup ordering. Actual SGLang MoE kernel compatibility and numerical quality require physical GPU validation and are not claimed from these tests.


## ADR-015: Fortress replaces obfuscation-first production inference

`docs/roadmap.md` supersedes the performance-first production architecture. Fortress outsources only exact quantized linear operations over additive integer shares and keeps nonlinear operations, attention mixing, routing, scheduling, and decoding at the trusted client combiner. Existing vLLM/SGLang/xDiT integrations remain plaintext numerical references and transition tools, not the Fortress production engine. A new Rust/CUDA engine is therefore required rather than another framework adapter.

The initial arithmetic domain is modulo `2^32`, with wide intermediates and a proof that every unreduced quantized product fits before execution. A prime-field alternative must be benchmarked and recorded before changing that domain. Full independently generated random additive shares can provide information-theoretic hiding under explicit non-collusion and independence assumptions. Seed-expanded shares reduce storage and bandwidth but provide only computational hiding. Algebraic transforms remain an inner layer and are never counted as a confidentiality guarantee.

Fortress has one security mode and does not silently fall back to plaintext, floating-point reference, or weaker transport execution. Every hardware ladder rung requires separate authorization and exact quantized, quality, attack, integrity, canary, cost, and teardown evidence. Independent attestation, second-party builds, external audits, and public red-team results remain external acceptance gates, not claims the implementation can make about itself.
