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

Provide provider credentials through the environment or the provider's secret mechanism. Do not put keys in YAML, shell history, process arguments, logs, manifests, or Git. Use a separate Ed25519 evidence-signing key at `/secure/evidence-signing.key` with mode `0600`; store only its public key at `/secure/evidence-signing.pub` for predecessor verification. Publish only that public key.

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

The final command validates 24 offline commands: 23 YAML ladder forms plus the real multi-model smoke command. It does not contact a provider or resolve unresolved artifacts.

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

GGUF graph semantics are accepted per `general.architecture` only (`llama`, `qwen2`, `qwen3`, `qwen3moe`, `gemma3`, `gemma4`; see `docs/models.md`). Before any GGUF rung, run the export differential on the exact files being deployed and keep its JSON with the rung evidence:

```bash
.venv/bin/python scripts/formats/gguf_export_differential.py /secure/models/model.gguf /secure/models/source-checkpoint \
  --tolerance 1e-6 --logits --output /secure/evidence/gguf-differential.json
```

Use `--tolerance` appropriate to the encoding (zero or 1e-6 for F32/F16/BF16; quantized encodings need the measured quantization error and are a quality measurement, not an equality check). A nonzero exit, a configuration difference, or a missing tensor blocks the rung. An architecture outside the list fails closed at `open_architecture`; adding one requires a new family entry justified by a passing differential, not a name-table edit. DeepSeek GGUF exports are not accepted; use the Safetensors checkpoint for L6.

The L4 dense selection is `Qwen/Qwen1.5-110B-Chat` under the Tongyi Qianwen license. L6 uses `deepseek-ai/DeepSeek-V3` under the DeepSeek model license and requires an operator-measured `verification.reap_saliency_path`. Its JSON maps each layer index to every original expert index and finite nonnegative saliency score (`configs/ladders/reap-saliency.example.json` shows only the schema, not production scores). Complete per-layer scores are mandatory. Pruning reads only retained experts, remaps them contiguously, restricts official/client routers consistently, and measures quality against the original unpruned model. Do not provision L6 with example or incomplete scores.

## 4. Required topology and resource preflight

A Fortress operation with `N` shares needs `N²` role cells for each of its activation and weight shares, plus any required replicas. Start with `N=2` and a separate trusted combiner. Hardware ladders generate a per-operation `PlacementMap` and package only each worker's assigned operations with operation-specific weight-share indices; an unplaced pool is rejected before SDK execution. The default 25% provider cap and anti-adjacency rules can require substantially more than four workers, especially for models with many operations. Local F0/M0 retain their explicit four-process reference topology and are not independent-host evidence. Operation placement enforces role uniqueness, anti-adjacency, query/key and gate/up/down separation, provider/country/CPU diversity, capacity, and randomized operation assignment.

Run `client.preflight()` before starting roles. Record:

- decoded logical parameters and source bytes;
- Fortress share storage and transient decoder memory;
- trusted client boundary and peak memory;
- per-role memory and aggregate memory per physical machine;
- expected activation/result bandwidth and relay mode;
- provider, country, CPU vendor, GPU model, VRAM, and independent machine IDs.

Connection count is a resource too. The trusted client opens one authenticated session, one QUIC connection, and one cover-traffic loop per operation per role, and a role worker's session limit scales with its assigned operations (four per operation, minimum 64). A dense 35-layer model has a few hundred operations; a 128-expert MoE has tens of thousands. Before L2, L5, and L6, record the operation count from `discover_fortress_operations`, the client's open-file limit (`ulimit -n`), thread count, and steady-state cover-traffic bandwidth, and raise the limits or reduce the operation set deliberately. This scaling has only been exercised up to roughly one hundred operations per worker on loopback.

Reject overcommit before role launch. Do not treat four roles on one machine as independent-host evidence. A relay forwards opaque datagrams; it is not an additional confidentiality layer.

Vast workers use the tracked provider state. Azure, Google, Alibaba, OCI, AWS, and IBM adapters are built into `ProviderRegistry.with_defaults()`; additional providers register a `ProviderAdapter` through the `xurcroh.providers` Python entry-point group. Teardown always requires `destroy`, `charges`, and `remaining` so cost and remaining-instance evidence cannot be skipped. A fully described external worker can omit a Vast state record but must supply `provider_instance_id`, `machine_id`, `provider_id`, `geolocation`, `cpu_vendor`, SSH connection fields, `public_host`, and `tls_port`. Unknown providers or incomplete verification fail closed. `startup_concurrency` defaults to 2 across uploads, discovery, role startup, and relay setup. V2 reports include live per-role CPU, RSS, and process-attributed GPU memory collected before shutdown.

Use only Azure NCC40ads H100 v5, Google A3 HighGPU TDX, or Alibaba gn8v-tee for a confidential-GPU test in this implementation. OCI E5/E6, AWS Nitro Enclaves, and IBM TDX are CPU-only paths and cannot satisfy a confidential-GPU gate. GCP G4 is not supported. Cloud API mocks are not provider evidence: record the actual region, shape, image digest, hardware measurements, firewall CIDRs, API responses, invoice currency, and independent destruction query for every physical provider run. AWS additionally requires an enclave-image EIF, expected PCRs 0/1/2, pinned root certificate, and an AF_VSOCK datagram relay; there is no SSH or external network inside the enclave. OCI requires a pinned Veraison verifier and expected measurements, Alibaba requires nonempty appraisal policy IDs and runtime binding, and IBM requires full Intel DCAP collateral. See `docs/deployment.md` for exact configuration inputs.

Run `scripts/fortress/tee_provider_smoke.py` once per selected provider before deploying a model grid. Its `--create` mode records a mode-`0600` recovery state, validates visibility and charge reporting, destroys the instance, and rejects any remaining instance. If the process exits after creation, rerun the same provider/offer with `--recover`; never create a replacement first. `configs/providers/README.md` gives exact commands. The lifecycle report deliberately leaves `physical_attestation_validated: false`; complete the provider-specific evidence gate separately. Charge queries occur before destruction in the grid teardown path, and a billing failure cannot prevent destruction.

## 5. Execution order

Complete each gate before spending on the next. Preserve signed evidence, raw JSON, actual worker exit codes, bandwidth, latency, attack results, cost ledger, and teardown proof.

0. **Theory gate (no provider):** every text and media manifest entry with a miniature checkpoint of its own model family through four loopback role workers and the real verifier. This is local evidence only and must be green on the delivered SHA before any spend:

   ```bash
   .venv/bin/python -m pytest -m slow tests/fortress/test_ladder_theory.py
   .venv/bin/python scripts/fortress/fetch_tiny_pipelines.py /secure/tiny-pipelines
   XURCROH_TINY_PIPELINES=/secure/tiny-pipelines .venv/bin/python -m pytest -m slow tests/fortress/test_media_ladder_theory.py
   ```

1. **F0/M0:** miniature text and DiT, four local processes, exact Fortress-reference outputs and measured traffic (`configs/ladders/f0-m0.yaml`).
2. **Format smoke:** Qwen3-0.6B GGUF, then EXL2 and EXL3 (`formats-smoke.yaml`, `exl2-l0.yaml`, `exl3-l0.yaml`).
3. **L0:** Gemma 4 E2B on four independent hosts (`gguf-l0.yaml` for the GGUF path).
4. **L1:** Gemma 4 E4B, then 12B.
5. **L2:** Gemma 4 26B A4B MoE; client routers, decoy expert calls, dropless padded dispatch.
6. **L3:** Gemma 4 31B and Qwen 3.8 27B, including VLM image/audio substeps where supported.
7. **M1:** small real image DiT, few steps.
8. **M2:** Flux and Qwen-Image, text-to-image, image-to-image, and masked editing.
9. **M3:** Wan 2.1 text-to-video (1.3B) and image-to-video (14B 480P). These are different checkpoints: a text-to-video denoiser takes 16 input channels and cannot run the image-to-video pipeline, which needs 36.
10. **M4:** Wan 2.2 two-denoiser text-to-video (T2V-A14B) and image-to-video (I2V-A14B), then MiniMax/Video DeltaNet recurrent state and supported audio/video paths. MiniMax H3 has no public miniature pipeline and has never executed locally; treat its first run as discovery.
11. **L4–L6:** only after resource and network measurement supports the selected 100–300B dense, large MoE, and approximately 1T REAP-pruned models.

For a single-rung manifest, the command shape is:

```bash
.venv/bin/python scripts/fortress/run_ladder.py configs/ladders/gguf-l0.yaml \
  --provisioned-config /secure/provisioned.json \
  --state /secure/provider-state.json \
  --output /secure/evidence/l0-gemma-4-e2b-gguf \
  --previous-evidence /secure/evidence/f0-tiny-llama/evidence.json \
  --evidence-public-key /secure/evidence-signing.pub \
  --signing-key /secure/evidence-signing.key
```

For `text-ladder.yaml`, `media-ladder.yaml`, or `f0-m0.yaml`, add `--model <manifest-model-name>`. The offline validator prints all 22 exact command forms. F0/M0 are roots; every later rung additionally requires `--previous-evidence <signed-predecessor.json>` and `--evidence-public-key <trusted-public-key>`. The runner verifies the predecessor's passing V2 result, rung order, and identical source revision before compilation or deployment. Use `--existing-deployment` only for a reviewed existing deployment, never together with `--provisioned-config`.

## 6. Numerical and output acceptance

Each rung requires:

- exact equality to the **Fortress quantized reference** (`quantized_reference_max_abs_error == 0`);
- separately measured source-model quality error, not a false bitwise-equivalence claim after source decoding plus Fortress requantization;
- text logits and sampled output checks, streaming and cancellation;
- VLM text/image/audio conditioning where implemented;
- exact local-versus-grid latent equality for image/video, plus deterministic PNG/MP4 and supported audio checks;
- non-square, odd-sized, and chunked temporal shapes for media;
- image-to-image, masked editing, and image-to-video workflow checks where the pinned pipeline exposes them. `verification.workflows` selects the required gates and `verification.workflow_pipelines` maps each to an installed Diffusers class; current manifests include Flux/Qwen image-to-image and inpainting and Wan image-to-video. Qwen-Image image-to-image uses `QwenImageImg2ImgPipeline`; `QwenImageEditPipeline` belongs to the separate Qwen-Image-Edit checkpoint and cannot load from `Qwen/Qwen-Image`. The verifier executes every selected workflow through local, remote, original, and public-SDK paths and rejects unsupported parameters rather than skipping them.

Retain model/output quality metrics and actual failures. Performance is measured, never a pass criterion. Text verification releases the compiled reference before loading the original model; media releases local/remote pipelines before original-model evaluation. V2 evidence includes per-phase timing and client RSS so large-model memory can be reviewed instead of assumed. If the original model does not fit alongside the trusted verifier after those releases, run the same pinned original-model quality comparison on a separate trusted reference host and join its signed result to the rung evidence; never skip or weaken that comparison.

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

Test multiple independent models, multi-user isolation, cancellation, timeouts, backpressure, worker SIGKILL/restart, connection interruption, packet delay/loss, stale sessions, and clean shutdown. Test aggregate-memory rejection before role launch. Preserve proof that all role processes and relay forwards terminate. The text ladder verifier exercises streaming, acknowledged cancellation, post-cancellation recovery, and concurrent public-SDK requests. Run the separate real multi-model gate with `python scripts/fortress/multi_model_smoke.py --config /secure/multi-model.json --output /secure/evidence/multi-model-smoke.json`; its configuration registers at least two real models and can use the managed SSH lifecycle.

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
.venv/bin/python scripts/fortress/container_smoke.py  # requires Docker; builds and exercises the sealed role
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
