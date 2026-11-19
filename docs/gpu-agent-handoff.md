# Funded GPU Validation Agent Handoff

This document is the public execution procedure for the next authorized GPU validation agent. It is not a completion or production claim. Physical L0–L6/M0–M4 evidence, independent-host execution, official-checkpoint quality, final CI, and production acceptance remain to be established.

## 1. Source and evidence rules

Record the delivered Git SHA before changing anything:

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline --decorate
```

- Deploy exactly a committed revision. The deployment scripts reject a dirty source tree and use `git archive HEAD`.
- Keep `requirement/`, internal directives, private work records, credentials, generated model shards, recipient keys, TLS keys, signing keys, prompts, source images, and decoded media out of the public repository and public evidence.
- Do not reuse older validation summaries as evidence for the delivered SHA.
- Label every result `implemented`, `locally_validated`, `hardware_validated`, or `production_accepted`. Only the first three may be established by these tests; production acceptance requires a separate acceptance decision.
- Failed and blocked runs are useful evidence. Record the exact command, failure, cleanup, and unverified follow-up; never turn them into passes.

## 2. Trusted environment and secrets

Use a trusted client/gateway for compilation, combination, tokenization, routers, normalization, attention state, sampling, schedulers, recurrent state, and VAEs. Untrusted workers receive only assigned encrypted/dispersed additive shares and random activation shares.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
./scripts/bootstrap-python.sh
.venv/bin/maturin develop --manifest-path crates/transport/Cargo.toml --features python
```

Provide provider credentials through the environment or the provider's secret mechanism. Do not put keys in YAML, shell history, process arguments, logs, manifests, or Git. Use a separate Ed25519 evidence-signing key at `/secure/evidence-signing.key` with mode `0600`; publish only its public key.

Verify before provisioning:

```bash
.venv/bin/python -m pytest -q -m 'not hardware and not slow' --cov --cov-report=term
.venv/bin/python -m ruff check python benchmarks tests examples
.venv/bin/python -m ruff format --check python benchmarks tests examples
.venv/bin/python -m mypy
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
.venv/bin/python scripts/fortress/validate_handoff.py
```

The final command validates 21 offline ladder commands. It does not contact a provider or resolve unresolved artifacts.

On resource-constrained hosts, run tests sequentially with an explicit process memory cap (for example `ulimit -v 5000000`) and persistent logs. Do not run managed video, multi-model, full-relay, or full-suite workloads concurrently. Do not retry the same expensive test blindly: preserve the log, identify the failing phase, and reduce the fixture or move to a larger host. Terminate only test-owned processes and confirm that no workers or relay forwards remain. Optimized official EXL reconstruction is selected automatically on CUDA hosts; exact CPU reference decoders remain available for local verification.

## 3. Checkpoint provenance and format gates

Every manifest with `resolution_required: true` must be completed before provisioning. For Hugging Face artifacts, verify the official repository, immutable 40-character commit, license, tokenizer, and complete checkpoint fingerprint. For local GGUF/EXL artifacts, additionally record source SHA-256, the original model's repository/revision/license, conversion tool and revision, and the pinned decoder identity from `ReferenceDecoderBridge`.

Do not provision while a manifest contains `REQUIRED_MODEL_SELECTION`, `REQUIRED_REVIEW`, an unresolved revision, or a zero final artifact fingerprint. `configs/ladders/README.md` explains the template fields.

Trusted checkpoint preparation uses:

```bash
.venv/bin/python scripts/fortress/prepare_checkpoint.py --help
.venv/bin/python scripts/formats/generate_gguf_fixtures.py /secure/fixtures/gguf
PYTHONPATH=.:python .venv/bin/python scripts/formats/generate_exl_fixtures.py /secure/fixtures/exl
```

Decode each generated fixture through `xurcroh.formats.open_checkpoint` before compiling. Physical EXL2/EXL3 runs must compare reconstructed matrices against the official pinned ExLlama extension. The local trellis fixtures are not a substitute for that differential check.

## 4. Required topology and resource preflight

A Fortress operation with `N` shares needs `N²` role cells for each of its activation and weight shares, plus any required replicas. Start with `N=2` and a separate trusted combiner. Hardware ladders generate a per-operation `PlacementMap` and package only each worker's assigned operations with operation-specific weight-share indices; an unplaced pool is rejected before SDK execution. The default 25% provider cap and anti-adjacency rules can require substantially more than four workers, especially for models with many operations. Local F0/M0 retain their explicit four-process reference topology and are not independent-host evidence. Operation placement enforces role uniqueness, anti-adjacency, query/key and gate/up/down separation, provider/country/CPU diversity, capacity, and randomized operation assignment.

Run `client.preflight()` before starting roles. Record:

- decoded logical parameters and source bytes;
- Fortress share storage and transient decoder memory;
- trusted client boundary and peak memory;
- per-role memory and aggregate memory per physical machine;
- expected activation/result bandwidth and relay mode;
- provider, country, CPU vendor, GPU model, VRAM, and independent machine IDs.

Reject overcommit before role launch. Do not treat four roles on one machine as independent-host evidence. A relay forwards opaque datagrams; it is not an additional confidentiality layer.

Vast workers use the tracked provider state. Other providers register a `ProviderAdapter` through the `xurcroh.providers` Python entry-point group; teardown requires its `destroy`, `charges`, and `remaining` implementations so cost and remaining-instance evidence cannot be skipped. A fully described external worker can omit a Vast state record but must supply `provider_instance_id`, `machine_id`, `provider_id`, `geolocation`, `cpu_vendor`, SSH connection fields, `public_host`, and `tls_port`. Unknown providers or incomplete verification fail closed. `startup_concurrency` defaults to 2 across uploads, discovery, role startup, and relay setup. V2 reports include live per-role CPU, RSS, and process-attributed GPU memory collected before shutdown.

## 5. Execution order

Complete each gate before spending on the next. Preserve signed evidence, raw JSON, actual worker exit codes, bandwidth, latency, attack results, cost ledger, and teardown proof.

1. **F0/M0:** miniature text and DiT, four local processes, exact Fortress-reference outputs and measured traffic (`configs/ladders/f0-m0.yaml`).
2. **Format smoke:** Qwen3-0.6B GGUF, then EXL2 and EXL3 (`formats-smoke.yaml`, `exl2-l0.yaml`, `exl3-l0.yaml`).
3. **L0:** Gemma 4 E2B on four independent hosts (`gguf-l0.yaml` for the GGUF path).
4. **L1:** Gemma 4 E4B, then 12B.
5. **L2:** Gemma 4 26B A4B MoE; client routers, decoy expert calls, dropless padded dispatch.
6. **L3:** Gemma 4 31B and Qwen 3.8 27B, including VLM image/audio substeps where supported.
7. **M1:** small real image DiT, few steps.
8. **M2:** Flux and Qwen-Image, text-to-image, image-to-image, and masked editing.
9. **M3:** Wan small text-to-video and image-to-video.
10. **M4:** Wan large multi-denoiser execution and MiniMax/Video DeltaNet recurrent state and supported audio/video paths.
11. **L4–L6:** only after resource and network measurement supports the selected 100–300B dense, large MoE, and approximately 1T REAP-pruned models.

For a single-rung manifest, the command shape is:

```bash
.venv/bin/python scripts/fortress/run_ladder.py configs/ladders/gguf-l0.yaml \
  --provisioned-config /secure/provisioned.json \
  --state /secure/provider-state.json \
  --output /secure/evidence/l0-gemma-4-e2b-gguf \
  --signing-key /secure/evidence-signing.key
```

For `text-ladder.yaml`, `media-ladder.yaml`, or `f0-m0.yaml`, add `--model <manifest-model-name>`. The offline validator prints all 21 exact command forms. Use `--existing-deployment` only for a reviewed existing deployment, never together with `--provisioned-config`.

## 6. Numerical and output acceptance

Each rung requires:

- exact equality to the **Fortress quantized reference** (`quantized_reference_max_abs_error == 0`);
- separately measured source-model quality error, not a false bitwise-equivalence claim after source decoding plus Fortress requantization;
- text logits and sampled output checks, streaming and cancellation;
- VLM text/image/audio conditioning where implemented;
- exact local-versus-grid latent equality for image/video, plus deterministic PNG/MP4 and supported audio checks;
- non-square, odd-sized, and chunked temporal shapes for media;
- image-to-image, masked editing, and image-to-video workflow checks where the pinned pipeline exposes them.

Retain model/output quality metrics and actual failures. Performance is measured, never a pass criterion.

## 7. Security and adversarial validation

Run the complete worker-compromise and transport suites for every format and media adapter. Give the adversary each worker's process environment, assigned decrypted artifacts, recipient private key, transport buffers, and labels. Test:

- no original checkpoint, source matrix, embedding/head/router, client basis, VAE, prompt, original media, or recoverable latent is available;
- no boundary key, peer shard key, other model state, or other worker's decryptable artifact is available;
- below-threshold collusion fails and full-grid collusion is reported as a residual risk, not hidden;
- stale-epoch mixing, replay, tampered manifests/shards, malformed parser inputs, and wrong attestation evidence fail closed;
- Freivalds failures are zero, canary attribution/revocation executes, and replica misbehavior is attributed;
- memory and logs contain no secrets, source prompts, or decoded outputs;
- access-pattern, timing, cache/TLB, power, and client-query leakage are measured and documented rather than claimed absent.

EXL2/EXL3 validation must include an official-extension differential run. GGUF validation must include all used type codes, shards, metadata arrays, and the applicable endian mode.

## 8. Reliability, recovery, and memory

Exercise direct QUIC and the datagram fallback independently. Measure bytes and time for compile, connection, each operation, queueing, kernel execution, and public SDK completion.

Test multiple independent models, multi-user isolation, cancellation, timeouts, backpressure, worker SIGKILL/restart, connection interruption, packet delay/loss, stale sessions, and clean shutdown. Test aggregate-memory rejection before role launch. Preserve proof that all role processes and relay forwards terminate.

Run on physical GPUs:

```bash
./scripts/build-native.sh
.venv/bin/pytest tests/hardware -q --runxfail
PYTHON=/path/to/cuda/python ./scripts/fortress/cuda_forensics.sh
```

Include exact field-matrix boundary products, seed expansion, zero-on-free/reallocation, KV/latent/recurrent/transport cleanup, and `compute-sanitizer --tool initcheck`. An epilogue does not prove all registers or SRAM were erased.

## 9. Attestation, release, and CI evidence

Where available, validate fresh AMD SEV-SNP, Intel TDX, and NVIDIA evidence against hardware-signed fields and current trust collateral. Bind nonce, workload/artifact identity, recipient key, and expected image measurement. Test secondary-device substitution and attestation failure. Commodity non-TEE execution remains a distinct mode, not silently equivalent attested execution.

Verify two clean wheel builds, SBOM/signature/transparency checks, ProVerif, protocol fuzzing, dependency audits, secret scanning, and GitHub Actions on the exact delivered SHA:

```bash
PATH=$HOME/.cargo/bin:$PATH CARGO_BUILD_JOBS=2 ./scripts/reproducible-build.sh
set -o pipefail; opam exec -- proverif protocol/fortress_session.pv | python3 scripts/check_proverif.py
.venv/bin/pip-audit
npm run check  # from apps/web-manual
```

Second-party reproducibility, confidential-hardware evidence, independent security review, and open red-team/bounty results remain separate evidence gates.

## 10. Cost control and mandatory teardown

Before every rung, record provider prices, GPU-hour estimate, gateway cost, checkpoint storage, image volumes, and inter-provider egress. Keep the watchdog active from provisioning until verified destruction. Uncapped spending, if authorized, still requires full accounting and TTL teardown.

On success or failure:

1. Stop workers and both relay stages; collect worker exit codes without collecting private prompts/media.
2. Destroy every gateway and worker instance using the tracked state.
3. Query the provider API independently for remaining instances and volumes.
4. Store the complete charge ledger and `remaining_instance_ids: []`.
5. Delete temporary checkpoint/shard/key material according to the secret-retention policy; retain only approved signed public evidence.

Stopping a process or instance does not eliminate storage charges. Do not declare teardown complete from local process output alone.

## 11. Final report contents

Return the exact branch/SHA, provider and GPU configurations, per-rung pass/fail/blocked results, source and decoder identities, exact quantized errors, source quality errors, bandwidth/latency/throughput, resource peaks, attack results, recovery results, CI links, reproducibility hashes, spending ledger, destruction proof, and remaining production blockers.

Do not declare universal future-model support, production-grade security, production acceptance, or full physical ladder completion without the corresponding direct evidence.
