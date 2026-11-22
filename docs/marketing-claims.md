# Marketing claims sheet

Status: designed for defence in depth; claims under evaluation.

This is the short form of `docs/claims.md` for people writing product copy. Each line gives
wording that may be used, what a customer can do to check it, and the sentence that has to appear
near it. If a claim is not on this page, do not make it. If the check in the middle column stops
working, stop using the claim. Evidence for each line is in `docs/claims.md` and
`docs/validation-report.md`.

## Headline

> **Don't just trust. Verify.** Every request comes back with a signed record of what ran and how
> it was protected, and you can check that record yourself, in your browser.

Must accompany it: "Xurcroh is designed for defence in depth. Its security claims are under
evaluation and have not been independently audited."

## Claims

### Trust of quality: you can check what ran

| Say | The customer's check | Say with it |
|---|---|---|
| "Every task has a signed trace: the model build, the settings, and how the links were encrypted." | `xurcroh verify trace.json`, or paste the trace into the Verify page | "When you run the client yourself. On the hosted API the trace is the operator's signed statement." |
| "Verify in your browser, with nothing installed and nothing uploaded." | Open the Verify page, disconnect from the network, paste a trace | "The page checks signatures and log proofs; it cannot check that a log still holds its history, which needs a live query." |
| "Each worker signs for the model pieces it loaded and the calls it answered." | The trace lists every worker signature; verification re-checks each one | "A signature identifies a key, not a machine or a location." |
| "Every worker result is checked mathematically before it is used." | The trace reports the check and its error bound | "The check is statistical: a wrong result passes with probability about 2^-64." |
| "A trace cannot be quietly changed later." | `xurcroh verify --proof ... --log-url ... --witness-key ...` | "Once its hash is in a public log that someone other than the log operator witnesses. We do not yet operate one." |
| "Only published model builds are served, and the client can refuse to run unless its own build was published." | `python -m xurcroh.build` prints the digest of an installation; compare it with the log | "The digest in a trace is the client's own report. Proof of what is installed comes from computing it yourself." |

### Privacy: who sees what

| Say | The customer's check | Say with it |
|---|---|---|
| "No single worker sees your prompt, your output, or the model." | Architecture and tests in the repository | "This holds if the workers do not pool what they hold." |
| "Cover traffic that a worker cannot tell from real requests." | `tests/fortress/test_cover_traffic.py` | "Per call. In the default mode a request still adds traffic on top of the cover rate." |
| "Optional constant-rate mode: one frame per time slot whether or not you are prompting." | Trace field `cover_mode` | "Off by default, and slow." |
| "Optional decoy requests at random times." | Trace field `decoy_tasks` | "Off by default." |
| "Workers can be reached through Tor." | Trace field `carrier` on each route | "This hides your address from the worker side. It does not make you anonymous, and it is slow: about 1 to 4 seconds per network round trip in our one measurement." |
| "The client records what it saw of each worker and can refuse workers that share a host or a key." | Trace field `observed` on each route | "Distinct hosts and keys do not prove distinct operators or countries; those remain the operator's declaration." |

### Cryptography

| Say | The customer's check | Say with it |
|---|---|---|
| "Hybrid post-quantum encryption (X25519 + ML-KEM-768) on every layer that protects prompts, activations and model shares." | The trace reports the negotiated group and session suite for each worker | "Hybrid: it holds if either scheme holds. TLS certificates are classical. Not independently reviewed." |
| "Hybrid post-quantum signatures (Ed25519 + ML-DSA-65) on traces, logs, releases and worker checkpoints." | Verification fails if either half fails | Same as above. |
| "Forward-secret sessions: a worker key stolen later does not open recorded traffic." | `protocol/*.pv` can be run with ProVerif | "Shown in a symbolic model of the handshake, not proved of the implementation." |
| "The session handshake has a machine-checked symbolic model, including the relayed route and the key ratchet." | Run ProVerif on the three model files | "A model of the protocol design. Do not say 'formally verified'." |

### Hosted API

| Say | The customer's check | Say with it |
|---|---|---|
| "Every hosted request returns a signed trace that only you can read." | Fetch `/v1/tasks/{id}/trace` with your key; another key gets nothing | "The hosted gateway sees your prompt and output. Its trace is the operator's statement about the model and settings, not proof." |

## Do not say

"Anonymous." "Quantum-proof" or "post-quantum secure" without "hybrid". "Formally verified."
"Hardware-attested." "Publicly logged." "Audited." "Zero-knowledge." "Homomorphic." "Military-grade."
"Unbreakable." "Most secure." Any speed or latency comparison. "Works over I2P" until a run on the
real I2P network passes.

## What would unlock stronger wording

| Stronger claim | What is needed | Who |
|---|---|---|
| "Publicly logged and independently witnessed" | A log at a public address, and a witness run by another organisation | Project management and a partner |
| "Independently audited" | An external cryptographic and code review | External auditor |
| "Hardware-attested workers" | Attestation wired into the role worker and run on confidential hardware | Engineering, with hardware |
| "Reproducible by anyone" | A second party reproducing the release build | A second party |
| Performance numbers | The authorized GPU ladder rungs | Project management authorization |
