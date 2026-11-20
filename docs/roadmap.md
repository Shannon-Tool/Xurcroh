# Xurcroh Fortress Roadmap

Status: direction document, written 2026-11-12 against revision `6a72cad`
(branch `fix/pre-gpu-corrective-implementation`). It tells the next engineer or agent what to build,
in what order, and what evidence each step needs. It supersedes the performance-first framing of the
original specifications wherever the two conflict.

Evidence labels used below: **confirmed** (read in the code), **suspected** (inferred from the code,
not yet reproduced), **estimate** (arithmetic, not measured), **published** (external paper or
advisory; verify against the primary source before relying on it).

## 0. Mission

**Make Xurcroh the most secure inference library that has ever existed.**

Decisions already made by the Project Manager:

- Xurcroh ships **one mode, Fortress**. There are no weaker or faster modes.
- **Weights are always mathematically masked**, for every model, public or private.
- **Slowness is acceptable.** Security wins every trade-off against speed. Speed is reported
  honestly and is never a pass criterion.
- Security is deliberately over-engineered: every independent layer is stacked.
- Text, image and video generation are all first-class.
- Scale target: a ~1T-parameter MoE; dense models up to 100–300B. The model ladder uses current
  models (Gemma 4 family, Qwen 3.8 27B) and existing compression (REAP, low-bit quantization).

Fortress in one sentence: no server ever holds a readable weight, prompt, activation, image or
output, and an attacker must defeat several independent layers at once.

What Fortress must beat, and how:

| Existing approach | Its limit | Fortress answer |
|---|---|---|
| Apple Private Cloud Compute | Single operator, custom hardware | The same properties (stateless, no privileged access, non-targetable, transparent) on rented commodity GPUs |
| GPU confidential computing (SEV-SNP/TDX + NVIDIA) | TEE.fail-class physical attacks extract attestation keys with an interposer costing under $1,000 (published); marketplace hosts have physical access | A TEE is one layer; the data inside it is already secret-shared |
| MPC / FHE | Too slow, because non-linear steps are computed under encryption | Only linear algebra is outsourced; the client does the cheap non-linear steps |
| Permutation / rotation schemes (current Xurcroh) | Broken by published attacks ("Hidden No More", ICML 2025) | Kept only as an inner layer under real secret sharing |

Required reading before design work: Gupta, Katz and Miers (IACR ePrint 2026/1849; client keeps
non-linear layers, two non-colluding servers compute linear layers on additive shares, with
verification) — the closest published design; Slalom (Tramèr and Boneh, 2019); Cascade
(arXiv 2507.05228); "Hidden No More" (ICML 2025); REAP (arXiv 2510.13999); AONT-RS (Resch and
Plank, FAST 2011); Whisper Leak (arXiv 2511.03675); TEE.fail; TOPLOC (ICML 2025).

## 1. Where the code is today

| Area | Reality at `6a72cad` |
|---|---|
| Hardware evidence | Qwen3-0.6B, FP32, ten shards run sequentially on one RTX 4090, one 9-token prompt, compared with the transformed FP64 reference rather than Hugging Face logits. Two-node numerics failed (errors 3.99 and 28.6) before later fixes and were not re-run. See `docs/validation-report.md`. |
| Privacy mechanism | Orthogonal conjugation with a per-model static basis. A worker holding the public checkpoint can align weights and recover the basis (suspected; the repo's own toy Procrustes test succeeds). For Qwen3 the query/key transform is exactly the identity (confirmed, `python/xurcroh/algebra/attention.py:72-78`). |
| Topology | Hub: the client relays hidden states between every pair of stages (`python/xurcroh/runtime/stage.py`, ADR-010). |
| Transport | The GPU run used HTTPS/JSON. Rust QUIC exists, but its listener always runs the FP64 CPU executor (`python/xurcroh/native_stage.py`). Rust onion routing is not exported to Python. |
| Engines | vLLM adapter hand-builds scheduler output behind a global lock and imports `unittest.mock` in production code (`python/xurcroh/adapters/vllm_v1_stage.py`). SGLang, MoE, VLM, image and video paths have never run on real models. |
| Reproducibility | The ten-stage harness lives only in gitignored `artifacts/vast/`, next to SSH, TLS and stage private keys. |
| Web manual | 22 sections of about 200 words each; the seven interactive labs are not mounted on any page. |

## 2. Stop-ship defects in code that Fortress will reuse

Fix these first. CPU only; no GPU spend. Each fix needs a regression test that fails on the current
code.

1. **AEAD key and nonce reuse across directions (confirmed).** `python/xurcroh/runtime/stage_sessions.py:45`
   and `:107` both build the sender with `(b"client", b"stage")`, and each side counts nonces from
   zero. Request *n* and response *n* are sealed with the same key and nonce. The same pattern is in
   `python/xurcroh/runtime/media_sessions.py` and `media_protected.py`. The controller relay sees
   both ciphertexts.
2. **Secret seed defaults to 0 (confirmed).** `python/xurcroh/cli.py:30`; the compiler uses
   `torch.Generator().manual_seed`. Use a CSPRNG and keep the secret in the client boundary only.
3. **Request state keyed by `request_id` alone (confirmed).** `python/xurcroh/models/dense.py:153`
   and the position map in `adapters/vllm_v1_stage.py`; the cancel and session-delete routes are
   unauthenticated (`python/xurcroh/stage_worker.py:326-334`).
4. **Controller has no authentication (confirmed)**; worker keys are self-asserted and regenerated
   on every start; the relay forwards to any registered endpoint (suspected SSRF)
   (`python/xurcroh/controller/service.py`).
5. **Attestation binding is not hardware-bound (confirmed).** The nonce is checked against a
   `report_data` string supplied alongside the evidence (`python/xurcroh/attestation.py:32-46`),
   never against the REPORT_DATA field inside the hardware-signed SNP report or TDX quote.
6. **Onion layer uses the raw X25519 output as the cipher key**, with fixed associated data and no
   padding (confirmed, `crates/crypto/src/onion.rs`).
7. **Stage private key placed in engine config objects** (confirmed,
   `python/xurcroh/adapters/vllm_v1_stage.py:312,339`).
8. **Hygiene.** Move the validation harness from `artifacts/vast/` into `scripts/gpu/`; rotate the
   keys stored there; add `.config/` to `.gitignore`; delete the stale `docs/requirements/matrix.yaml`
   (the tracked `docs/requirements-matrix.yaml` is current); fix the dead references in `README.md`,
   `CONTRIBUTING.md` and `SECURITY.md`; open a pull request to `main`.

## 3. Fortress specification

### 3.1 Arithmetic

Everything outsourced to a server is integer arithmetic in the prime field `2^32-5` (the prime just
below `2^32`, chosen so field elements still fit 32-bit words). Weights are quantized to about 4
bits and activations are fixed-point at 8–12 bits, so true results fit the field. In this field a
uniformly random mask is a true one-time pad: hiding is
information-theoretic, and Freivalds checks are mathematically sound. Rescaling after each product
is done by the client in the clear, which avoids the hardest part of general MPC.

### 3.2 The share grid

The compiler splits every weight matrix into random shares `W = W_1 + … + W_n`. The client splits
every activation into random shares `x = x_1 + … + x_n`. Server role `(i, j)` holds `W_j`, receives
`x_i`, and returns `W_j · x_i`. The client adds the n² results to obtain `W · x`. Servers never talk
to each other and no dealer is needed.

- Each server sees one uniformly random weight share and one uniformly random activation share. It
  learns nothing about the model or the prompt, even with unlimited computing power.
- Reading a prompt requires all n activation-share roles for one weight share to collude. Reading
  the weights requires all n weight-share roles to collude. Start with n = 2 (four roles); keep the
  design general for n = 3.
- `W_1` is expanded from a short seed on the GPU (start from the ChaCha20 kernel in
  `native/cuda/otp_mask_kernels.cu`), so only `W_2 = W − W_1` is stored. Benchmark seed expansion
  against storing both shares.
- Shares are refreshed every epoch by adding and subtracting a fresh random matrix, so shares stolen
  at different times cannot be combined.

### 3.3 How weights are split across servers

Seven layers, all mandatory:

| Layer | Rule | Purpose |
|---|---|---|
| At rest | Each shard is AEAD-encrypted, then dispersed with an all-or-nothing transform plus Reed–Solomon coding (AONT-RS) across many storage nodes. The key is released by threshold from independent holders after attestation. | No storage node holds a decodable piece; node loss is survivable |
| Share | The grid in 3.2 | No server holds a readable weight |
| Granularity | Split by operation, not by layer: each attention projection, each feed-forward matrix and each expert is a separate unit | Finer scatter |
| Placement | (a) No server holds two operations that are adjacent in the computation. (b) No server or provider holds more than a set fraction of the model. (c) Never both halves of a pair: query with key, gate with up/down, expert with its router. (d) Expert units are scattered pseudo-randomly under a secret permutation. (e) The placement map is secret, client-held, and reshuffled every epoch. | Collusion yields fragments; access patterns are unlinkable across epochs |
| Client-only tensors | Embeddings, output head, first and last block, every MoE router, normalisation gains, modulation tables, text encoders, VAE | The small tensors that define behaviour never leave the client |
| Diversity | The roles of one grid cell sit with different providers, countries and CPU vendors (AMD and Intel). At least two replicas per unit and role. | One subpoena, one vendor break or one outage is not enough |
| Inner layer | The existing algebraic transforms (`python/xurcroh/algebra/`) are applied before sharing | Full collusion reconstructs only a rotated model |

Compression order: REAP expert pruning → algebraic transform → quantize → share.

### 3.4 The trusted combiner

The client, or a gateway machine the user owns, holds the placement map, deals activation shares,
adds results, and runs every non-linear step in the clear: normalisation, attention mixing with its
own KV cache, activation functions, MoE routing, sampling and schedulers. It never needs the large
weights.

Bandwidth, not compute, is the cost: every linear operation is a round trip carrying full activation
shares. Measure this in simulation before any GPU spend. Optimisations to evaluate: deriving one
share from a seed so only the other is transmitted; pre-computing the random-side products before
the prompt exists; batching several operations per round. A later research item removes the combiner
by running the non-linear steps under multi-party computation between share holders.

### 3.5 Integrity

For every weight share the compiler gives the client a short check vector. The client runs a
Freivalds check on every returned product (reuse and correct `python/xurcroh/integrity.py`, which is
currently unused and loses exactness in floating point) and re-executes a random fraction on replica
servers. A wrong answer is detected and attributed to a server.

### 3.6 Mixture-of-Experts

Routers stay on the client, so servers never see routing logic. A server cannot tell a real expert
call from a dummy, because inputs are random shares; the client therefore pads every token to a
fixed number of expert calls with decoys. Expert placement is reshuffled each epoch. Measure what the
remaining access pattern leaks.

### 3.7 Image and video generation

Diffusion needs Fortress most: a worker that sees a latent can decode it with the public VAE and
look at the picture, with no inversion attack required.

- Text encoder, noise generation, scheduler, guidance mixing and VAE encode/decode stay on the client.
- Every DiT linear layer (patch embedding, attention projections, feed-forward, modulation
  projections) runs on the grid. The combiner does normalisation, modulation and attention.
- Conditional and unconditional guidance passes use different server sets. Image-to-image and
  image-to-video inputs are shared like any other latent. Recurrent video state stays at the combiner.
- Bandwidth per image is very large (thousands of tokens × hundreds of operations × tens of steps,
  estimate), so the combiner must sit on a fast link to the servers: a user-owned gateway by default.
  Start with few-step distilled models.

### 3.8 Surrounding layers (all mandatory)

- **Cryptography.** Noise or HPKE handshakes in place of hand-rolled sessions; forward secrecy on
  both sides; hybrid post-quantum key exchange (`X25519MLKEM768` in rustls); mutual TLS; key rotation
  wired in (`crates/crypto/src/session.rs:80`, currently never called); the handshake modelled in
  Tamarin or ProVerif.
- **Anonymity.** The client reaches the network through Tor/I2P or an oblivious relay; servers never
  learn who they serve and talk only to the client. Fixed-size frames and constant-rate cover
  traffic; Whisper Leak shows padding alone is insufficient, so measure classifier accuracy.
- **Decoys and honeytokens.** Real requests are mixed with decoys. Some carry canaries (fake
  credentials, tracking URLs). A tripped canary proves a break, names the servers that saw it, and
  revokes their keys.
- **Prompt armour.** A small local model replaces names and secrets with placeholders before
  sharing and restores them in the output.
- **Hardware.** Every server in a confidential VM with fresh attestation bound to the
  hardware-signed report, wherever such hardware is available.
- **Sealed worker image.** Minimal, immutable, measured; no shell or SSH; engine sandboxed.
- **Stateless servers.** Scrub VRAM and host memory after every call; exclusive GPU tenancy; no
  request data in logs; replace the always-true assertion at
  `tests/hardware/test_cuda_forensics.py:28` with a scrub-then-probe test.
- **Transparency.** Reproducible builds, signed releases, SBOM, and a public log of worker-image
  measurements that clients check.

### 3.9 How earlier ideas fit

| Idea | Verdict |
|---|---|
| Encrypt safetensors and spread them over many servers | Yes, twice: AONT-RS at rest, and the share grid while computing |
| New custom inference engine | Yes, required: vLLM and SGLang cannot compute on integer shares. They remain only as plaintext references for numerical tests. Record this as an ADR; it departs from Deliverable A in the master directive. |
| New custom kernel | GPU: yes, integer matrix kernels and seed expansion. Operating system: a sealed minimal image, not a new kernel. |
| VM in VM | A nested VM does not stop the host; use a confidential VM plus an inner sandbox |
| I2P + Tor | Yes, for the client's path into the network |
| Obfuscation | Kept as the inner layer; never counted as a guarantee |
| Honeypot | Yes: decoys and canaries |

### 3.10 Residual risks to state openly

All roles of one grid cell colluding; a compromised client; access-pattern and timing leakage; a
client learning weights it queries (matters only for private models); denial of service. The attack
suite (extend `python/xurcroh/attacks.py`, which is toy-scale today) must measure each one.

### 3.11 Evidence required before using the words "most secure"

1. Section 2 fixed, with regression tests.
2. Attack-suite results published for every ladder rung.
3. Handshake formally modelled; fuzzing running in CI.
4. A run on real confidential hardware with captured evidence.
5. A reproducible build verified by a second party; transparency log live.
6. An external cryptography and systems audit, with findings published.
7. A public per-threat comparison against Private Cloud Compute, TEE services and MPC.
8. A bounty or open red-team period with no unresolved critical finding.

Until then, the README and manual say "designed for defence in depth; claims under evaluation".

## 4. Engineering work

1. **Fortress engine** (new, Rust + CUDA): integer matrix kernels, seed expansion, share storage,
   stateless request handling. Benchmark int32 kernels against exact floating-point modular
   arithmetic (Slalom's approach). Check kernel choices against current CUDA documentation rather
   than assumptions.
2. **Compiler**: extend `python/xurcroh/shards/compiler.py` (already streams per layer) with
   quantization, share generation, check vectors, per-operation units and AONT-RS output. Reuse the
   shard format and recipient sealing in `python/xurcroh/shards/format.py` and `recipients.py`.
3. **Client combiner**: extend `python/xurcroh/client/boundary.py` and `python/xurcroh/runtime/stage.py`
   into the full non-linear executor with KV cache, routers, schedulers and Freivalds checks.
4. **Transport**: make Rust QUIC (`crates/transport`) the only data plane; replace the 64 MiB frame
   cap (`crates/protocol/src/frame.rs:6`) with chunked streams; add mutual authentication.
5. **Placement service** on the client; the controller is reduced to authenticated discovery.
6. **Provisioning**: lift the two-node cap (`scripts/vast/provision.py:161`); add multi-provider
   support for the diversity rule; keep the budget guard and watchdog.
7. **Model support**: Gemma normalisation handling (suspected bug around
   `python/xurcroh/shards/compiler.py:183`), MLA, shared experts, and real diffusers loaders in place
   of the bespoke DiT format (`python/xurcroh/shards/dit_compiler.py`).
8. **Tests, CI and manual**: integer-exact end-to-end tests; cargo-fuzz; container build and scan;
   mount the seven manual labs and add share-grid, placement and Freivalds labs; correct sections
   08, 15 and 22; update `docs/requirements-matrix.yaml`.

## 5. Sizing (estimates to verify)

A masked weight costs about 4 bytes regardless of quantization. With n = 2 and seed expansion, two
stored copies are needed; without it, four. Assumes about 70 GB usable per 80 GB GPU and 20 GB per
24 GB GPU.

| Model | Parameters shared | Stored (2 copies) | 80 GB GPUs | 24 GB GPUs |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | a few billion | ~20–40 GB | 2 | 2–4 |
| Gemma 4 12B | 12B | ~100 GB | 2 | 5–6 |
| Gemma 4 26B MoE / 31B, Qwen 3.8 27B | ~30B | ~250 GB | 4 | 12–13 |
| Dense 100–300B | 100–300B | 0.8–2.4 TB | 12–35 | 40–120 |
| ~1T MoE, REAP 50% | ~0.5T | ~4 TB | ~58 | ~200 |

Add the seed-expanding roles (compute only) and replicas. The 1T rung is expensive by design; the
small rungs prove everything first at low cost.

## 6. Ladders

Each rung is a gate. GPU spend requires Project Manager authorization per rung. A rung passes only
with: exact match to the quantized reference, quality versus the original model, attack-suite
results, Freivalds and canary tests, a cost ledger and teardown proof.

| Rung | Text model |
|---|---|
| F0 | Existing miniature models on CPU, four local processes: full Fortress end to end |
| L0 | Gemma 4 E2B on four independent hosts (Qwen3-0.6B kept as a regression) |
| L1 | Gemma 4 E4B, then 12B |
| L2 | Gemma 4 26B A4B (first MoE: routers on the client, decoy expert calls) |
| L3 | Gemma 4 31B and Qwen 3.8 27B |
| L4 | Dense 100–300B class |
| L5 | Large MoE (200B–700B class) |
| L6 | ~1T MoE, REAP-pruned |

| Rung | Media model |
|---|---|
| M0 | Existing synthetic DiT (`python/xurcroh/models/dit.py`) on CPU: Fortress end to end; bandwidth measured |
| M1 | Smallest current real image DiT, few-step |
| M2 | Flux and Qwen-Image; image-to-image and editing |
| M3 | Wan small text-to-video |
| M4 | Wan large and the MiniMax / Video DeltaNet path |

Run the text path first for multimodal Gemma and Qwen models. Verify exact Hugging Face repository
IDs, revisions and licences before each rung, and prefer the newest open model of each class if the
specification's choice has been superseded.

## 7. Order of work

1. Section 2 defects.
2. F0 and M0 on CPU: proves the mathematics and measures bandwidth.
3. Engine, compiler, combiner and transport (section 4, items 1–4).
4. L0.
5. Surrounding layers (3.8).
6. L1–L3 and M1–M2.
7. A run on confidential hardware.
8. L4–L6 and M3–M4.
9. The evidence gate (3.11).
