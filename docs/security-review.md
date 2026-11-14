# Internal Security and Systems Review

**Scope:** Current Fortress corrective branch through `c3bea8e`, with subsequent documentation-only updates. This is an internal engineering review, not an external audit, formal proof, or production certification.

## Review method

- Trace client, worker, provider, relay, and attestation trust boundaries in `docs/security.md` and `docs/architecture/decisions.md`.
- Exercise the complete local suite: 374 passed, 23 deselected, 81.59% branch coverage at `c3bea8e`; the previously passed long video ladder remains separate.
- Review arithmetic forgery, noncanonical encoding, signed-word conversion, replay, wrong-key, cross-stage key isolation, worker-memory, and artifact-tampering regressions.
- Verify 29 Rust tests, Clippy/formatting, RustSec, Python/npm audits, and 32 frontend interaction tests.
- Inspect physical evidence and teardown records rather than treating executable tests as hardware results.

## Corrected findings

| Finding | Correction and regression evidence |
|---|---|
| Ring/prime disagreement and truncated native arithmetic | Unified `2^32-5` field; canonical transport and forgery/truncation regressions |
| Signed CUDA result conversion shifted residues by five | Shared unsigned canonicalization in controller and worker; CPU and physical native tests |
| Provider manifests lost SSH proxy settings | SSH/TLS endpoint separation tests in `tests/fortress/test_ladder_provider.py` and `test_ladder_ssh.py` |
| Worker launch and certificate generation were incompatible with the Rust transport | DER identities, real local mutual-TLS handshake, detached/background SSH launch tests |
| Provider failures could bypass gateway teardown | Outer smoke lifecycle wrapper and failure-path teardown regression |
| Direct provider UDP was unavailable | Optional loopback-only SSH datagram relay with real local QUIC coverage; no physical pass claimed |
| Idle relay connections could fail permanently | One bounded reconnect for connection/timeout errors with the same stateless role request |
| Documentation retained the removed modular-ring design | Prime-field decision and evidence updated in `docs/architecture/decisions.md` |
| Release verification did not check SBOM content or constrain artifact paths | Version-2 manifests verify the SBOM digest and reject path/symlink escapes; deterministic SBOM UUIDs and a standalone verification CLI are tested |

## Per-threat comparison

| Threat | Xurcroh design | Single-operator private-cloud / TEE approach | Conventional MPC |
|---|---|---|---|
| Worker host reads process/GPU memory | Partitioned custody, additive shares, and optional TEE layers reduce exposure; obfuscation alone is not semantic security | Hardware isolation is the primary boundary and depends on vendor/operator trust | Cryptographic shares are the primary boundary under a non-collusion threshold |
| Workers collude | Full role collusion reconstructs protected values; placement/diversity raises the attack cost but is not a proof | Collusion is typically outside the single-provider model | Explicit threshold assumption; exceeding it breaks confidentiality |
| Provider or network intercepts traffic | Mutual authenticated QUIC, directional session keys, replay protection, and optional onion/SSH relay | Provider-controlled encryption and network boundary | Authenticated channels plus MPC protocol-specific protections |
| Malicious worker returns incorrect results | Field challenges, canaries, and quarantine detect tested misbehavior; no universal floating-point proof is claimed | Trusted hardware/attestation and operator controls | Verifiable or robust MPC variants may provide stronger guarantees at additional cost |
| Hostile physical access or attestation compromise | Optional TEEs add a layer but do not eliminate physical trust-root risks | Directly exposed to vendor implementation, firmware, interposer, and trust-root risks | Cryptographic model does not depend solely on TEE hardware, but endpoints remain vulnerable |
| Global timing/cache/TLB/power observation | Fixed frames, cover traffic, and dispatch padding are mitigations only; leakage remains | Hardware/provider side-channel controls remain necessary | Protocol communication and computation patterns may also leak metadata |
| Client or release infrastructure compromise | Not prevented by worker protections; local secrets and signing roots remain critical | Same fundamental endpoint and supply-chain risk | Same fundamental endpoint and supply-chain risk |

## Residual limitations and approval boundary

No local test can establish non-collusion, physical-jurisdiction independence, complete GPU register/SRAM erasure, universal model quality, side-channel immunity, or production availability. The independent multi-host numerical/SDK gate remains open. Provider credit/capacity currently blocks further allocation, and hosted CI is externally blocked before job execution.

Within that boundary, the internal reviewer finds no unresolved locally reproducible critical defect in the tested implementation paths. This statement does not waive the hardware, security-review, model-quality, or operational acceptance gates.
