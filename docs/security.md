# Security Assessment

## Protection boundaries

- Client boundary: embeddings, LM head, raw prompts/media, final sampling, and decoded media remain client-owned. `ClientBoundary` wipes its tensor store on close.
- Model custody: each worker receives only an encrypted, transformed stage and separate wrapped key. Plaintext monolithic loading is rejected in strict mode.
- Transport: pinned-certificate QUIC, directional HKDF keys, ChaCha20-Poly1305, nonces, replay windows, rotation, and onion-style hop metadata protect transit. The controller relays opaque bytes without payload keys. Protected remote image/video sessions additionally use ephemeral X25519 agreement and directional AEAD tensor envelopes; workers return transformed latents rather than decoded media. Strict media uses compiled equivariant DiT stages whose requests contain no basis or spatial-permutation secrets; inverse transformation and VAE decode occur on the client. Legacy worker-side latent recovery is explicitly development-only `transport_only` obfuscation and is not a zero-trust guarantee. Source-image editing performs VAE encoding on the client, and video reconstruction/MP4 encoding occurs on the client.
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

CPU tests cover cache/request cleanup, sealed memfds, nonce/binding failures, finite-field and floating-point checks, Procrustes recovery, norm leakage, and real model equivalence. CUDA tests are executable and skip cleanly without CUDA. At `a42b552`, one RTX 4090 passed 20 native/hardware tests and 18 `compute-sanitizer` tests with zero reported errors; this is scoped native-path evidence, not complete physical-memory erasure or independent multi-host acceptance. Full distributed-model, TEE, side-channel, and performance claims remain hardware-validation work.


## Fortress prompt armour and canaries

`xurcroh.fortress.prompt_armour` detects explicit credentials and email addresses locally and can run a token-classification model strictly from a local checkpoint with network loading disabled. Recognized spans are replaced with random-session placeholders and restored in generated text. Recognition coverage is model-dependent and is not a guarantee that every name or secret is found. Placeholder originals are held in wipeable bytearrays, but Python strings, caller-owned inputs, generated strings, and framework copies cannot be comprehensively zeroized by this API.

`xurcroh.fortress.canary` mixes real and decoy call schedules and associates unique share canaries with their observer set. A trip means that a registered secret was observed by the trusted monitoring integration; it attributes the observer set and triggers revocation, but does not prove which observer disclosed it. The F0/M0 reports use a simulated external observation of an actually transmitted share, not an independent monitoring service or a public red-team result.


## HPKE session establishment

Interactive stage and protected-media sessions use RFC 9180 HPKE (`X25519HkdfSha256` with ChaCha20-Poly1305) to seal a fresh random session master. Stage HPKE associated data binds the session identifier and authentication token; media HPKE associated data binds the media session identifier. Directional AEAD keys, replay windows, and monotonic epoch rotation are derived only after that authenticated establishment. The legacy raw-X25519 session-exchange path has been removed. Offline shard recipient sealing is a separate artifact-custody mechanism, not this interactive handshake.


## Oblivious relay and traffic analysis

`xurcroh.fortress.relay` forwards mutually authenticated QUIC calls through an allowlisted oblivious relay. Onion layers reveal only the current hop's authorized next address; HPKE and directional AEAD remain end-to-end between client and worker. The relay observes the client's network address, and each worker observes the relay's network address. A global observer can still correlate timing and volume, so this is not Tor/I2P-level anonymity or proof against intersection attacks. Fixed-size frames and constant-rate verified cover calls reduce direct size/workload signals but do not remove scheduling or microarchitectural leakage. The current local leave-one-out nearest-mean timing result is recorded in `docs/evidence/fortress/transport.json` (balanced accuracy 0.572 versus a 0.5 baseline for that run), not presented as general classifier resistance.

## SSH datagram relay fallback

The optional `datagram_tunnel` deployment path forwards framed UDP datagrams through SSH local port forwarding when provider UDP is unusable. Its TCP and UDP listeners bind to loopback, each client source receives an isolated relay connection, and worker/relay PIDs are included in teardown. QUIC mutual TLS, application session authentication, and payload encryption remain end-to-end; SSH is an additional transport layer, not a replacement for either. The relay does not provide anonymity against SSH hosts, providers, global observers, or traffic analysis. Local tests demonstrate a real QUIC round trip, while the physical relay run did not complete its numerical/SDK gate and is not claimed as validated.
