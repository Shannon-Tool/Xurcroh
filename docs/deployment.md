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

Select `engine="fortress"` and set `fortress_shares=N` for per-operation `N × N` grids. Managed deployment requires at least `3N²` SSH hosts, truthful `provider.country` and `provider.cpu_vendor` metadata, and enough aggregate capacity for the default 25% per-provider operation cap; small models with awkward operation counts can require additional hosts. The public client automatically discovers memory, resolves a pinned hub checkpoint when `allow_download=true`, compiles recipient-sealed role artifacts, deploys the committed source archive, bootstraps the locked CUDA environment using the Fortress-only `requirements-fortress-cuda.lock` subset rather than installing vLLM/SGLang on every role. The runtime package exports `RuntimeService` lazily so role-worker imports do not pull unrelated HTTP, media, SSH, or model-framework dependencies into that minimal environment. It then generates deployment-scoped mutual TLS identities, and starts `xurcroh.fortress.worker_cli` over authenticated QUIC. The CA private key is never written to the SDK workspace, and each recipient private key is deleted locally after its successful upload; only the client identity and public recipient keys remain for runtime use. No global model key or plaintext checkpoint is sent to workers. Each model receives isolated role packages and ports on shared hosts. Use the client as a context manager so owned role processes, local private artifacts, and remote deployment directories are cleaned up. Shutdown waits for role exit and force-stops stuck roles. If SSH cleanup fails, the SDK retains the local recovery workspace and raises `ClusterUnavailableError`; calling `close()` again retries cleanup without discarding the recovery state. The complete Gemma E2B example is in `README.md`; resource checks are available through `client.preflight()` and `await AsyncXurcroh.preflight()`.

Provisioning requires a clean committed source checkout so the remote archive exactly matches the client revision. It rejects non-Fortress models. Automatic provisioning supports text decoders and Diffusers `transformer`/`unet` components; Gemma/Qwen text execution does not deploy unused vision/audio towers. Existing initialized Fortress endpoints remain supported for separately managed deployments. Shared grid transport uses one connection and encrypted session per role, per-session key rotation, bounded session ownership, and grid-rate cover traffic rather than per-operation threads and connections. For NAT-restricted workers, `datagram_tunnel=true` starts per-model SSH datagram forwards and tracks local forwarders and remote relay daemons through the same teardown/recovery lifecycle. Managed discovery, uploads, role startup, and relay setup use `startup_concurrency=2` by default so a small trusted gateway does not create an unbounded process/SSH burst; operators can tune this explicitly for measured gateway capacity.

`deploy/fortress-worker.Dockerfile` builds the sealed `xurcroh.fortress.worker_cli` image with the minimal `requirements-fortress.lock` runtime, root Cargo workspace, matching Python 3.12 interpreter/native ABI, no final shell or SSH, and UDP/9444. `deploy/fortress-worker.yaml` supplies explicit Fortress arguments, a read-only artifact secret, writable scratch, non-root/read-only security settings, and Python-native readiness. `scripts/fortress/container_smoke.py` builds the image, executes an exact field product through mutual-TLS QUIC, checks a clean exit, and removes the container. CI also records the image identity, emits a CycloneDX SBOM, and runs Trivy. Local Docker execution is not claimed on hosts without a container toolchain.

## Confidential cloud provider adapters

Install `./scripts/bootstrap-python.sh` to include the `cloud` dependency extra. `ProviderRegistry.with_defaults()` registers Azure, Google Cloud, Alibaba Cloud, OCI, AWS, and IBM in addition to entry-point plugins. Every adapter exposes offer discovery, creation, destruction, remaining-instance checks, and charge reporting. APIs are called through official SDKs or authenticated HTTPS endpoints; the local tests use API doubles and do not establish cloud availability or hardware evidence.

| Provider | Enabled platform | GPU boundary | Required deployment inputs and verification |
|---|---|---|---|
| Azure | `Standard_NCC40ads_H100_v5`, AMD SEV-SNP + H100 NVL 94 GiB | Confidential GPU | `AZURE_SUBSCRIPTION_ID`, resource group, SSH public key, explicit `allowed_source_ranges`; verify composite CPU/GPU evidence before key release |
| Google Cloud | `a3-highgpu-1g`, Intel TDX + H100 80 GiB | Confidential GPU | Application Default Credentials, `GOOGLE_CLOUD_PROJECT`, SSH key, explicit source CIDRs, and `GOOGLE_BILLING_EXPORT_TABLE` for parameterized BigQuery cost queries |
| Alibaba Cloud | `gn8v-tee`, Intel TDX + H100 96 GiB per GPU | Confidential GPU | ECS/BSS credentials, `ALIBABA_REGION_ID`, approved Alibaba Cloud Linux 3 image, existing key pair/VPC/vSwitch, source CIDRs, and nonempty `ALIBABA_ATTESTATION_POLICY_IDS` |
| OCI | `VM.Standard.E5.Flex` or `VM.Standard.E6.Flex`, AMD SEV-SNP | **CPU only; no confidential-GPU claim** | OCI config, tenancy/compartment OCIDs, image, subnet, scoped NSGs, and a pinned Veraison verifier public key with expected measurements |
| AWS | Nitro Enclaves on a supported parent, initially `m5.xlarge` | **CPU only; enclave has no GPU or external networking** | AWS credentials, AMI/key pair/subnet/scoped SGs, pinned Nitro root certificate, expected PCRs 0/1/2, and an AF_VSOCK relay |
| IBM VPC | Current profiles advertising `tdx`, Intel TDX | **CPU only; no confidential-GPU claim** | `IBM_ACCOUNT_ID`, IBM credentials, image/VPC/subnet/scoped SGs/SSH key IDs, and Intel DCAP collateral |

Google G4 is not registered as an equivalent path because its AMD evidence flow is different and has no adapter here. CPU TEE instances must not be presented as protected GPU workers. All providers require independent physical verification of the exact region, image, hardware, evidence chain, firewall, cost API, and destruction behavior before acceptance.

For Azure, Google, and Alibaba, `XURCROH_ALLOWED_SOURCE_RANGES` can supply comma-separated CIDRs when `allowed_source_ranges` is not passed explicitly; there is no wildcard default. GCP billing uses bound query parameters and validates the export table identifier. Alibaba rejects non-approved images and incomplete OIDC policy/runtime binding. OCI follows the installed SDK's `platform_config.is_memory_encryption_enabled=True` model on E5/E6 SNP shapes, verifies a nonce-bound Veraison EAR with a pinned ES256 key, and requires affirming hardware/runtime-opacity claims and expected measurements. AWS verifies COSE_Sign1/ES384, a pinned root-first CA bundle, freshness, PCRs, nonce, and enclave public key before requesting KMS decryption. `NitroParentRelay` preserves length-framed opaque UDP datagrams over AF_VSOCK and accepts only a configured peer; HPKE/mTLS/application cryptography remains end-to-end inside the enclave and is never terminated on the parent. IBM delegates to the existing Intel DCAP verifier with full collateral and workload binding.

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

Use a secret manager for private identities and rotate them through the recipient distribution process. Start the remaining stages with distinct recipient identities and endpoints; use `--engine sglang` for the SGLang MoE path. The public model definition uses `path` for the local tokenizer/config, `manifest_path` for the compiled manifest, `boundary_key_env` for indirect client-only key supply, and `engine="vllm"` or `engine="sglang"`. Workers validate recipient identity, checkpoint fingerprint, manifest digest, tensor descriptors, and AEAD/replay state. QUIC clients use `quic://worker-a:9445`, `labels.ca_path`, and `labels.server_name`; `labels.attestation_endpoint` selects the HTTPS evidence endpoint. Native connections use per-worker locks, graceful shutdown, and timeout-triggered abort. For TEE mode, supply public evidence with `--attestation-evidence /run/xurcroh/attestation.json`; clients request a fresh challenge, verify nonce-bound evidence locally, and invoke the AMD/Intel verifier. NVIDIA evidence can use the local verifier or nonce-bound NRAS verification; composite CPU/GPU policy is required for confidential-GPU mode. Never put private attestation keys in this JSON document. `engine="reference"` is development-only FP64 numerical verification; optimized engine compatibility and numerical quality still require physical GPU validation.

Strict mode rejects full-model remote checkpoints even over HTTPS. Development policy is required for the ordinary full-model OpenAI-compatible worker path. Local `local://cpu` execution remains a trusted-client capability.

## Client configuration

Use `benchmarks/cluster.example.yaml` as a schema example. Set the worker credential indirectly through the named environment variable, install the trusted CA in the client, and use an HTTPS endpoint. A local model path is used for the client boundary/model definition; remote model names are not silently redirected. Multiple workers are alternates in the public remote chat path; algebraic pipeline-stage transport must be selected by the stage runtime rather than assumed from a worker list.

## Shutdown and failure behavior

Clients should be context managers. Closing cancels active work, waits for in-flight engine calls to finish cancellation cleanup, and then releases model and transport resources. Idle open streams do not block shutdown. A bounded shutdown timeout leaves resources available for a `close()` retry rather than tearing them down underneath active execution. Workers close their runtime at FastAPI lifespan shutdown. Registration expiry removes unhealthy workers from controller topology. Request admission is bounded per model. These local lifecycle tests do not prove recovery from physical host failure or WAN partition.

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
