# Deployment and Operations

Xurcroh is undergoing physical hardware validation. These files provide reproducible service packaging; packaging and scoped single-GPU results are not evidence of independent multi-host acceptance, production security acceptance, or measured production performance.

## Build

```bash
./scripts/build-wheels.sh
docker build --tag xurcroh:0.1.0 .
```

The first command creates distinct `xurcroh` and `xurcroh-quic` wheels in `dist/`. Install both for SDK and native transport support; `CARGO_BIN` can select a rustup toolchain not already on `PATH`. The multi-stage `Dockerfile` installs the locked Python environment, compiles the Rust transport extension with Rust 1.99.1, and runs as UID/GID 10001. It does not contain checkpoints, credentials, generated shards, or CUDA memory libraries. Build `native/cuda` for the target GPU architecture as described in `docs/hardware-validation.md`. The locked development environment uses CPU PyTorch; a GPU image must use the checkpoint-compatible PyTorch/CUDA wheel set and record that lockfile before deployment. See `docs/resources.md` for analytical GPU topologies and hardware risks.

## Loopback integration environment

`deploy/compose.yaml` is explicitly a development configuration. It uses unencrypted HTTP inside one Compose network and binds published ports to loopback. It is not a zero-trust production topology.

```bash
export XURCROH_MODEL_ROOT=/absolute/path/containing/tiny
# From the repository root:
docker compose --project-directory . -f deploy/compose.yaml up --build
curl http://127.0.0.1:9442/health
curl http://127.0.0.1:9443/health
```

The model root must contain `tiny/` with a Transformers-compatible checkpoint. Replace that name and model argument for a real checkpoint. Stop services with `docker compose -f deploy/compose.yaml down`. Do not pass credentials on a command line or store them in Compose files.

## TLS worker and controller

A standalone worker requires a certificate, key, and immutable local model path:

```bash
.venv/bin/python -m xurcroh.worker \
  --worker-id worker-a \
  --model qwen-7b=/models/qwen-7b \
  --host 0.0.0.0 --port 9443 \
  --tls-cert /run/xurcroh/tls/tls.crt \
  --tls-key /run/xurcroh/tls/tls.key \
  --controller https://controller.example:9443 \
  --advertised-endpoint https://worker-a.example:9443
```

Start the controller with:

```bash
.venv/bin/uvicorn xurcroh.controller.service:create_controller_app --factory \
  --host 0.0.0.0 --port 9443 \
  --ssl-certfile /run/xurcroh-controller/tls/tls.crt \
  --ssl-keyfile /run/xurcroh-controller/tls/tls.key
```

`--advertised-endpoint` is the address other nodes use; `--host` is only the bind address. Workers refresh signed registration every 20 seconds. Private identity keys remain in the worker process. Configure certificate verification and hostnames correctly; disabling verification is not a supported strict-mode fallback.

`/health` reports liveness only. `/metrics` and `client.metrics()` expose aggregate request counters and total duration, not prompts, tokens, media, credentials, or secrets. Metrics are not differential-privacy guarantees; restrict their network visibility.

## Kubernetes review checklist

`deploy/kubernetes.yaml` defines one controller, two independently addressable GPU workers, TLS secret mounts, a read-only checkpoint PVC, non-root/read-only containers, GPU requests, and namespace network policy. Before applying it:

1. Replace `CHECKPOINT_STORAGE_CLASS` and the image name with actual infrastructure values.
2. Provision `xurcroh-controller-tls`, `xurcroh-worker-a-tls`, and `xurcroh-worker-b-tls` using an approved CA. Do not commit private keys.
3. Populate the checkpoint PVC through an authenticated artifact process; verify immutable checkpoint revisions and licenses.
4. Select nodes with the intended GPU and high-bandwidth topology. Keep tensor/expert groups within a high-bandwidth host group; use pipeline stages across independent hosts.
5. Add an explicit ingress policy for the authorized client/controller networks. The supplied namespace-only policy intentionally denies other ingress by default.
6. Configure attestation and strict encrypted-shard loading before treating workers as zero-trust stages. Merely running two copies of a plaintext checkpoint is not the Xurcroh zero-trust architecture.
7. Run `docs/hardware-validation.md` and real numerical/security acceptance tests. The repository does not include a completed independent multi-host production-acceptance result.

Validate YAML locally with the deployment integration tests. `kubectl apply` is a Stage Two infrastructure action and has not been run.

## SDK-managed Fortress grids

For the current Fortress text path, provide four SSH-configured CUDA hosts for the default two-share grid and select `engine="fortress"`. The public client automatically discovers memory, resolves a pinned hub checkpoint when `allow_download=true`, compiles recipient-sealed role artifacts, deploys the committed source archive, bootstraps the locked CUDA environment, generates deployment-scoped mutual TLS identities, and starts `xurcroh.fortress.worker_cli` over authenticated QUIC. No global model key or plaintext checkpoint is sent to workers. Each model receives isolated role packages and ports on shared hosts. Use the client as a context manager so owned role processes, local private artifacts, and remote deployment directories are cleaned up. Shutdown waits for role exit and force-stops stuck roles. If SSH cleanup fails, the SDK retains the local recovery workspace and raises `ClusterUnavailableError`; calling `close()` again retries cleanup without discarding the recovery state. The complete Gemma E2B example is in `README.md`; resource checks are available through `client.preflight()` and `await AsyncXurcroh.preflight()`.

Provisioning requires a clean committed source checkout so the remote archive exactly matches the client revision. It does not automatically provision non-Fortress models or multimodal diffusion towers. Existing initialized Fortress endpoints remain supported for separately managed deployments. Shared grid transport uses one connection and encrypted session per role, per-session key rotation, bounded session ownership, and grid-rate cover traffic rather than per-operation threads and connections.

## Encrypted dense stages

Compile the checkpoint on a trusted host with a separate client-boundary key and one X25519 recipient identity per worker. Distribute only each stage's `.xurcroh` shard, its recipient-sealed key envelope, and that worker's private identity. Retain the boundary shard and client key on the trusted client. Start each worker with its own identity file and an optimized engine:

```bash
.venv/bin/python -m xurcroh.stage_worker \
  --manifest /models/compiled-qwen/manifest.json --stage-index 0 \
  --stage-private-key /run/xurcroh/stage-0.identity \
  --engine vllm --checkpoint /models/qwen \
  --host 0.0.0.0 --port 9444 \
  --tls-cert /run/xurcroh/tls/tls.crt --tls-key /run/xurcroh/tls/tls.key \
  --quic-listen 0.0.0.0:9445 \
  --quic-cert /run/xurcroh/tls/tls.crt --quic-key /run/xurcroh/tls/tls.key
```

Use a secret manager for private identities and rotate them through the recipient distribution process. Start the remaining stages with distinct recipient identities and endpoints; use `--engine sglang` for the SGLang MoE path. The public model definition uses `path` for the local tokenizer/config, `manifest_path` for the compiled manifest, `boundary_key_env` for indirect client-only key supply, and `engine="vllm"` or `engine="sglang"`. Workers validate recipient identity, checkpoint fingerprint, manifest digest, tensor descriptors, and AEAD/replay state. QUIC clients use `quic://worker-a:9445`, `labels.ca_path`, and `labels.server_name`; `labels.attestation_endpoint` selects the HTTPS evidence endpoint. Native connections use per-worker locks, graceful shutdown, and timeout-triggered abort. For TEE mode, supply public evidence with `--attestation-evidence /run/xurcroh/attestation.json`; clients request a fresh challenge, verify nonce-bound evidence locally, and invoke the AMD/Intel verifier. NVIDIA remote evidence is not yet supported. Never put private attestation keys in this JSON document. `engine="reference"` is development-only FP64 numerical verification; optimized engine compatibility and numerical quality still require physical GPU validation.

Strict mode rejects full-model remote checkpoints even over HTTPS. Development policy is required for the ordinary full-model OpenAI-compatible worker path. Local `local://cpu` execution remains a trusted-client capability.

## Client configuration

Use `benchmarks/cluster.example.yaml` as a schema example. Set the worker credential indirectly through the named environment variable, install the trusted CA in the client, and use an HTTPS endpoint. A local model path is used for the client boundary/model definition; remote model names are not silently redirected. Multiple workers are alternates in the public remote chat path; algebraic pipeline-stage transport must be selected by the stage runtime rather than assumed from a worker list.

## Shutdown and failure behavior

Clients should be context managers. Closing cancels active work and releases model resources. Workers close their runtime at FastAPI lifespan shutdown. Registration expiry removes unhealthy workers from controller topology. Request admission is bounded per model. These local lifecycle tests do not prove recovery from physical host failure or WAN partition.

## Release and transparency verification

Create a deterministic CycloneDX SBOM and sign release artifacts with a protected Ed25519 key:

```bash
.venv/bin/python scripts/release.py dist \
  --project . --sbom dist-sbom.json --signing-key /secure/release-ed25519.key
```

Verify the resulting version-2 manifest on a trusted host:

```bash
.venv/bin/python scripts/verify_release.py dist/release-manifest.json \
  --public-key <release-public-key-hex>
```

Version 2 binds the SBOM path and digest, validates every artifact digest, and rejects absolute paths, parent traversal, and symlink escapes. The verifier requires the referenced SBOM to exist at its recorded trusted path. Store signing keys outside the release directory and never commit them.

A transparency service can be run with `python -m xurcroh.transparency_cli` using an indirect write-token environment variable, persistent storage, and TLS certificate/key options. Clients must pin the Ed25519 public key and required image measurements; `scripts/verify_transparency.py` verifies the signed hash chain. A publicly hosted log is an external deployment gate and is not claimed from local tests.
