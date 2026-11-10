# Security Assessment

## Protection boundaries

- Client boundary: embeddings, LM head, raw prompts/media, final sampling, and decoded media remain client-owned. `ClientBoundary` wipes its tensor store on close.
- Model custody: each worker receives only an encrypted, transformed stage and separate wrapped key. Plaintext monolithic loading is rejected in strict mode.
- Transport: pinned-certificate QUIC, directional HKDF keys, ChaCha20-Poly1305, nonces, replay windows, rotation, and onion-style hop metadata protect transit. The controller relays opaque bytes without payload keys. Protected remote image/video sessions additionally use ephemeral X25519 agreement and directional AEAD tensor envelopes; workers return latents rather than decoded media. Source-image editing performs VAE encoding on the client, and video reconstruction/MP4 encoding occurs on the client.
- Memory: page-locked shard staging, sealed memfds, request cleanup, physical KV-slot wipes, native zero-on-free, and CUDA masking/scrubbing reduce residual data. Logical and hardware cleanup are distinguished.
- Optional TEE: AMD SEV-SNP, Intel DCAP, and NVIDIA verifier interfaces validate freshness plus workload/artifact/public-key binding. Strict, TEE, non-TEE, and development policies are explicit. A deployment may require one or a configured set of supported trust roots; AMD and Intel are not incorrectly required on the same host. Remote stage workers issue fresh challenges and clients verify nonce+workload+artifact+public-key-bound evidence locally, with AMD/Intel hardware verifier adapters. Remote NVIDIA report transport remains unsupported and fails explicitly rather than invoking an unrelated local verifier. No silent downgrade is allowed.

## What is not guaranteed

Orthogonal and contragredient transformations are computational obfuscation, not FHE, secret sharing, or a zero-knowledge proof. They may preserve norms, inner products, attention patterns, or other invariants. Public-weight Procrustes and norm attacks are included to demonstrate this boundary. Colluding workers, malicious computation, endpoint metadata, global timing, power/cache/TLB channels, physical bus attacks, and compromised attestation roots require separate mitigations.

PRG-based masking is not an information-theoretic one-time pad. Constant-load MoE padding makes packet sizes uniform but does not eliminate timing, TLB, cache, power, or implementation side channels. The uniform page-touch path is a mitigation to measure, not a proven elimination.

Independent sanitizer kernels cannot prove every physical register or SRAM location is erased. Python/framework copies and host root/hypervisor access limit physical-erasure claims. dm-verity verifies stored artifacts, not runtime computation.

Finite-field Freivalds checks have their standard algebraic domain. BF16/FP16/FP8 verification uses tolerances and empirical differential checks, not the finite-field probability bound.

## Attestation trust roots

The AMD adapter invokes `snpguest verify attestation`. Intel uses the `sgx_dcap_quote_verify` API with full collateral and expiration. NVIDIA invokes `python3 -m verifier.cc_admin` with a fresh nonce and rejects failure/devtools output. These interfaces do not establish immunity to physical PCK extraction, quote replay, forged CPU/GPU binding, malicious firmware, or a compromised verifier host. Deployments must pin trust roots and review current vendor guidance.

## Local evidence

CPU tests cover cache/request cleanup, sealed memfds, nonce/binding failures, finite-field and floating-point checks, Procrustes recovery, norm leakage, and real model equivalence. CUDA tests are executable but skipped without CUDA. Full physical multi-GPU, TEE, side-channel, and performance claims remain Stage Two validation.
