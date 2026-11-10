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
