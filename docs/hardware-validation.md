# Multi-GPU and Hardware Validation

No physical GPU results are asserted in the pre-GPU handoff. The following programs are ready for the Project Manager's Stage Two infrastructure.

## Native build and forensics

```bash
./scripts/build-native.sh
.venv/bin/pytest tests/hardware -q --runxfail
```

The suite builds the allocator and ChaCha20 mask, exercises shared-memory scrubbing and mask/unmask round trips, and requires `nvcc` plus a CUDA device. Without those prerequisites it reports two explicit skips.

Validate separately:

1. Allocator counters and zero-on-free for realistic model allocations.
2. KV block canaries before/after request completion and reallocation.
3. `compute-sanitizer --tool initcheck` on all native paths.
4. GPU architecture-specific shared-memory/local-memory behavior. An epilogue is not proof of complete register/SRAM erasure.
5. Constant-load MoE timing, cache, TLB, and power side channels; do not use packet size as the sole privacy metric.

## TEE and artifact validation

- AMD: install `snpguest`, generate a fresh report, verify against the deployment certificate set, and bind the report data to workload/artifact/public key.
- Intel: install `sgx-dcap-quote-verify-python`, supply current PCK/TCB/QE collateral and expiration, and pin the expected trust root.
- NVIDIA: install `nv-local-gpu-verifier`, run `python3 -m verifier.cc_admin --nonce <fresh-hex>`, reject devtools, and bind CPU/GPU evidence.
- Package encrypted shards with EROFS/dm-verity, bind expected root hashes, and test corrupt disk reads.

Physical PCK extraction, quote forgery, secondary-GPU substitution, and relay/host compromise must be treated as adversarial test cases, not assumed impossible.

## Cluster validation

Use at least two independent GPU workers for dense pipeline checks and a suitable high-bandwidth group for TP/EP. Test real checkpoint outputs, worker failure/rescheduling, cancellation, packet delay/loss, multi-user isolation, memory teardown, and WAN/NAT behavior. Record actual prefill/inter-token latency, throughput, security overhead, memory, and image/video quality in machine-readable benchmark output. Do not use the analytical 40–50 ms/token target as a measured result.

## Reproducible CUDA package

`deploy/cuda/Dockerfile` builds the pinned vLLM 0.15.0 / PyTorch 2.9.1 / CUDA 12.8 runtime from `requirements-cuda.lock`. `deploy/cuda/compose.yaml` describes two isolated stage containers with one GPU each. Docker and NVIDIA Container Toolkit are required; the development host used for CPU validation does not provide them, so an image build is not itself evidence of GPU execution.

Download a checkpoint explicitly on a trusted host before compilation. The initial validation target is the reviewed `Qwen/Qwen3-0.6B` architecture; record the exact revision and license with the result. Do not use a floating model revision as evidence.

```bash
./scripts/gpu/compile_checkpoint.sh /trusted/models/Qwen3-0.6B /trusted/artifacts/qwen3-compiled 2
```

The compiler creates independent recipient identities, a client-only boundary key, and three distribution classes:

- `client/` and `client-boundary.xurcroh` remain on the trusted client.
- `stage-N/` contains only the manifest, one encrypted stage shard, and its recipient envelope. `identities/stage-N.key` is copied separately only to worker N.
- `worker-checkpoint/` contains configuration/tokenizer metadata and no safetensors. vLLM needs this metadata, but workers must never receive the original plaintext checkpoint.

For two physical hosts, copy `configs/two-node.example.json`, replace its addresses and trusted local paths, and run:

```bash
./scripts/gpu/deploy_two_node.sh /trusted/config/two-node.json
.venv/bin/python scripts/gpu/verify_two_node.py /trusted/artifacts/two-node-validation/deployment.json
```

Deployment refuses a dirty repository and uses `git archive HEAD`, so the remote source matches the recorded revision. It starts both distributed rendezvous participants, exposes only TLS worker endpoints, and waits for optimized-runtime capabilities. The verifier then uses `from Xurcroh import Xurcroh` for synchronous, streaming, concurrent, cancelled, and post-cancellation requests. It writes a JSON report with actual response metadata and marks production acceptance false.

The deployment script stops worker processes if its own startup fails. Provider-instance destruction and independent billing verification are separate mandatory steps; killing a process or stopping an instance does not eliminate storage charges.

Aggregate the SDK report, actual `hardware.json` files, provider ledger, and destruction evidence with:

```bash
.venv/bin/python scripts/gpu/report.py \
  --sdk-report /trusted/artifacts/two-node-validation/sdk-report.json \
  --hardware /trusted/artifacts/worker-a-hardware.json \
  --hardware /trusted/artifacts/worker-b-hardware.json \
  --ledger /trusted/artifacts/vast-ledger.json \
  --destruction /trusted/artifacts/vast-destruction.json \
  --output /trusted/artifacts/validation-report.json
```

A successful SDK report establishes hardware execution of that request path only. It does not establish MoE/media execution, complete side-channel resistance, numerical equivalence across every model, or production acceptance.
