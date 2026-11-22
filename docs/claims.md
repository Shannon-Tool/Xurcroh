# Claims and their evidence

Status: designed for defence in depth; claims under evaluation.

This page lists the security properties the system provides today. Each is stated in the form
given here, and only while its "Check it yourself" column works. Local tests show an implementation
behaves as described on one machine. They are not hardware validation, an external audit, or a
guarantee. The roadmap's external evidence gates (independent attestation, second-party
reproducible build, external audit, public comparison, bounty period) are not complete, so no
superlative such as "most secure" may be used.

## Claims that can be made now

| Wording | What it rests on | Check it yourself | Condition that must be stated |
|---|---|---|---|
| "No single worker sees your prompt, the model weights, or the output." | Weights and activations are additively secret-shared over a prime field; each role worker holds one weight share and receives one activation share | `tests/fortress/test_f0.py`, `tests/security/test_worker_compromise.py` | Holds only if workers holding complementary shares do not collude |
| "Post-quantum key exchange at every layer between you and a worker (hybrid X25519 + ML-KEM-768)." | TLS key exchange (rustls `prefer-post-quantum`), the role session handshake and the onion layers each combine X25519 with ML-KEM-768; signatures combine Ed25519 with ML-DSA-65 | `cargo test -p xurcroh-transport --test quic`, `tests/fortress/test_role_session.py`, `tests/fortress/test_onion.py`; each task trace reports the group its worker links negotiated and the session suite | "Hybrid": security rests on either the classical or the post-quantum scheme holding. TLS certificates are classical, which affects live impersonation only. No NIST known-answer vectors are checked in. Applies to Fortress role workers, not the older stage and media sessions |
| "Forward-secret sessions: a worker key stolen later does not open recorded traffic." | The session key depends on a secret sealed to an ephemeral client key that is discarded; keys then move through a one-way ratchet | `tests/fortress/test_role_session.py`; `protocol/fortress_role_session.pv` (ProVerif: secrecy holds after the worker's static key is disclosed) | The formal model is symbolic and covers direct routes. Python cannot guarantee discarded keys are erased from memory. Through relays the session is not bound to a TLS connection |
| "Every worker result is mathematically checked before it is used." | Two independent Freivalds challenges per weight share, held only by the client | `tests/fortress/test_freivalds_rounds.py`, `tests/fortress/test_transport.py` | A wrong result is accepted with probability about 2^-64 per check; the bound is statistical |
| "Every task has a signed trace you can verify." | Trace of submitted and effective settings, model revision and source digest, sharing and transport facts, with prompt and output commitments | `xurcroh verify trace.json --public-key KEY --messages M --output O` | With the local client you check the model and settings yourself; through the hosted gateway they are the operator's statement. See `docs/task-trace.md` |
| "Each worker signs for the model artifacts it loaded and the traffic it answered." | Worker identity key (Ed25519 with ML-DSA-65); signed session checkpoint checked by the client and stored in the trace | `xurcroh verify trace.json` prints the number of verified worker checkpoints; `tests/fortress/test_checkpoint.py` | Identifies a signing key, not a location or hardware. Not attestation |
| "Hosted API: every request returns a task ID and a signed trace only you can read." | Gateway assigns the task ID, restricts trace and proof to the API key that made the task, and signs each trace as hosted and operator-asserted | `tests/gateway/test_gateway.py`; `xurcroh verify` prints the assurance | Must be described as operator-asserted. The gateway sees prompts and outputs in the clear. It shows the trace was not altered afterwards, not that the stated model or settings were used |
| "A trace cannot be quietly changed after it is issued." | Trace hash appended to a Merkle log; inclusion proof against a signed tree head | `xurcroh verify trace.json --proof proof.json --log-key KEY` | True beyond your own process only when a public log is configured (`security.trace_log_endpoint`) and checked with `--log-url`. State who operates the log, and claim independent witnessing only if a party other than the operator runs a witness and you verify with `--witness-key`. The project does not operate a log or witness |
| "Only model builds published in the log are served." | `security.require_logged_manifest` checks the model manifest digest against the public log before any output is returned | `tests/runtime/test_task_trace.py` | Covers the model manifest, not the client software build. Someone must have checked the build before publishing its digest |
| "Seeded generation is reproducible, and unseeded generation tells you the seed it used." | One generator per request; effective seed in `execution.seed` and in the trace | `tests/runtime/test_sampling.py` | Same model, settings, software version and hardware arithmetic |
| "Traffic is sent in fixed-size frames with constant cover traffic." | 16 KiB frames; cover calls at a fixed interval | `docs/evidence/fortress/transport.json` | Reduces size and workload signals. Does not hide endpoints, global timing, or volume |
| "Worker traffic can be carried over Tor or I2P." | SOCKS5 and SAM v3 carriers with QUIC, TLS and the session kept end to end; one Tor circuit identity per worker; carrier recorded in the trace | `tests/fortress/test_carriers.py` | You run Tor or I2P yourself, and the worker operator publishes the onion service or I2P tunnel. Hides your address from the worker side, not from an observer of both ends. Much slower than direct (seconds per round trip in the one real-Tor run). Tor has passed a transport test on the real network; I2P has run only against a stand-in, so do not claim I2P as working until it has |
| "Optional onion routing hides your address from workers." | Layered encryption through allowlisted relays, each layer sealed with X25519 and ML-KEM-768 | `tests/fortress/test_relay.py`, `tests/fortress/test_relay_cli.py` | Needs at least two relays run by non-colluding parties. Not Tor or I2P. A global observer can correlate timing |
| "Workers are spread across countries and hardware vendors." | Placement requires at least two countries and two CPU vendors per operation grid | `tests/fortress/test_placement.py` | The labels are declared by the operator and are not verified |
| "Reproducible, signed releases." | Double wheel build with digest comparison; release manifest signed with Ed25519 and ML-DSA-65; CycloneDX SBOM | `scripts/reproducible-build.sh`, `scripts/verify_release.py` | Not yet reproduced by a second party |
| "Secrets are kept out of core dumps and debuggers." | Client and role worker disable core files and mark the process non-dumpable | `tests/security/test_process_protection.py` | Does not stop a root user or hypervisor on the same host |
| "Optional prompt armour removes emails and credentials before the model sees them." | Local placeholder substitution, restored in output, reported per request | `tests/fortress/test_prompt_armour.py` | Off by default. Recognition is not exhaustive, and it changes the text the model reads |

A further true statement worth using: the secret sharing itself does not rely on a mathematical
problem that a quantum computer is expected to solve. That applies to uniformly random shares under
non-collusion. Seed-expanded shares, which are the default for activations, are computational.

## Claims that must not be made yet

| Wording to avoid | Why | What would make it true |
|---|---|---|
| "Post-quantum secure" or "quantum-proof", unqualified | The schemes are hybrid and unaudited here; TLS certificates and the older stage, media and shard-sealing paths are classical | An external cryptographic review, post-quantum certificates, and conversion or removal of the older paths |
| "Perfect forward secrecy", unqualified | Shown by tests and a symbolic model, not a computational proof; key erasure in Python is best effort | A computational analysis and key handling moved fully into memory-safe zeroizing code |
| "Anonymous over Tor/I2P", or any speed claim for those modes | The carriers hide your address from the worker side only; timing and volume can still be correlated, and latency has not been measured | Published traffic analysis and measured latency on the real networks |
| "Anonymous" | A global observer can correlate timing; one relay sees both ends | Multi-party relays or a real anonymity network, with published analysis |
| "Verified on your hosted request" | The hosted gateway's traces are the operator's statement; an unattested gateway could misreport the model or settings | An attested gateway whose trace key is bound to its attestation |
| "Hardware-attested workers" | Fortress role workers run unattested on commodity GPUs | Attestation wired into the role worker and validated on real confidential hardware |
| "Publicly logged" | The trace log is in-process | A public log service with independent witnesses |
| "Zero-knowledge", "homomorphic encryption" | Neither technique is used | Not planned |
| "Military-grade", "unbreakable", "most secure" | Not measurable, and the external evidence gates are open | External audit, comparison and bounty gates in the roadmap |
| "Formally verified" | The ProVerif model has three queries and assumes authentication | A model covering authentication, key compromise and replay |

## Performance wording

Each model layer is a network round trip to four workers, and the client checks every result. This
is much slower than ordinary inference. No speed claim may be made until hardware rungs publish
measured numbers.
