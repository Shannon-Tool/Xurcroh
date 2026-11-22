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
| "Post-quantum key exchange on worker links (hybrid X25519 + ML-KEM-768)." | rustls with `prefer-post-quantum` on mutual-TLS QUIC | `cargo test -p xurcroh-transport --test quic`; each task trace reports the group its worker links negotiated | Key exchange only. Session, onion and storage layers are classical; signatures are hybrid (Ed25519 with ML-DSA-65) |
| "Every worker result is mathematically checked before it is used." | Two independent Freivalds challenges per weight share, held only by the client | `tests/fortress/test_freivalds_rounds.py`, `tests/fortress/test_transport.py` | A wrong result is accepted with probability about 2^-64 per check; the bound is statistical |
| "Every task has a signed trace you can verify." | Trace of submitted and effective settings, model revision and source digest, sharing and transport facts, with prompt and output commitments | `xurcroh verify trace.json --public-key KEY --messages M --output O` | Local client only today. See `docs/task-trace.md` for what a trace does not establish |
| "Each worker signs for the model artifacts it loaded and the traffic it answered." | Worker identity key (Ed25519 with ML-DSA-65); signed session checkpoint checked by the client and stored in the trace | `xurcroh verify trace.json` prints the number of verified worker checkpoints; `tests/fortress/test_checkpoint.py` | Identifies a signing key, not a location or hardware. Not attestation |
| "A trace cannot be quietly changed after it is issued." | Trace hash appended to a Merkle log; inclusion proof against a signed tree head | `xurcroh verify trace.json --proof proof.json --log-key KEY` | The log is inside the client process today, not public and not independently witnessed |
| "Seeded generation is reproducible, and unseeded generation tells you the seed it used." | One generator per request; effective seed in `execution.seed` and in the trace | `tests/runtime/test_sampling.py` | Same model, settings, software version and hardware arithmetic |
| "Traffic is sent in fixed-size frames with constant cover traffic." | 16 KiB frames; cover calls at a fixed interval | `docs/evidence/fortress/transport.json` | Reduces size and workload signals. Does not hide endpoints, global timing, or volume |
| "Optional onion routing hides your address from workers." | Layered encryption through allowlisted relays | `tests/fortress/test_relay.py`, `tests/fortress/test_relay_cli.py` | Needs at least two relays run by non-colluding parties. Not Tor or I2P. A global observer can correlate timing |
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
| "Post-quantum secure" or "quantum-proof", unqualified | Only the TLS key exchange is post-quantum | Hybrid ML-KEM in the session, onion and storage layers |
| "Runs over Tor" or "I2P" | No Tor or I2P carrier exists | The planned carriers, with measured latency |
| "Anonymous" | A global observer can correlate timing; one relay sees both ends | Multi-party relays or a real anonymity network, with published analysis |
| "Verified on your hosted request" | No hosted gateway exists, and an unattested gateway could misreport | An attested gateway whose trace key is bound to its attestation |
| "Hardware-attested workers" | Fortress role workers run unattested on commodity GPUs | Attestation wired into the role worker and validated on real confidential hardware |
| "Publicly logged" | The trace log is in-process | A public log service with independent witnesses |
| "Zero-knowledge", "homomorphic encryption" | Neither technique is used | Not planned |
| "Military-grade", "unbreakable", "most secure" | Not measurable, and the external evidence gates are open | External audit, comparison and bounty gates in the roadmap |
| "Formally verified" | The ProVerif model has three queries and assumes authentication | A model covering authentication, key compromise and replay |

## Performance wording

Each model layer is a network round trip to four workers, and the client checks every result. This
is much slower than ordinary inference. No speed claim may be made until hardware rungs publish
measured numbers.
