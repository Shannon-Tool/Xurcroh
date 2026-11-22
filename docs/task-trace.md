# Task traces

Status: designed for defence in depth; claims under evaluation.

Every completed chat, image, and video request produces a signed trace on the trusted client. The trace is addressed
by the task's request ID (`response.execution.request_id`) and states what was asked for, what ran,
and how it was protected. It contains no prompt or output text, only salted commitments to them.

```python
response = client.chat.completions.create(model="m", messages=messages, temperature=0.7)
trace = client.tasks.trace(response.execution.request_id)
```

```bash
xurcroh verify trace.json --public-key <hex> --messages messages.json --output output.txt
```

`verify` exits non-zero and names the failing part if the signature does not verify, if the trace
was signed by a different key, or if the supplied prompt or output is not the one committed to.

## Contents

| Section | Fields |
|---|---|
| `request.settings` | Every setting as submitted. Chat: model, temperature, top_p, top_k, max_tokens, stop, seed, stream, timeout. Image and video: model, size, steps, strength, frames, fps, seed, timeout |
| `request.effective` | The seed that actually drove sampling (drawn if none was given), the sampler version, the number of prompt-armour substitutions |
| `request.prompt`, `result.output` | Salted SHA-256 commitments to the canonical messages and to the output text |
| `result` | Finish reason, token counts, duration |
| `model` | Name, revision and whether it is pinned, source format, digest of the source checkpoint files, layout digest, architecture, encodings, the repository name when the checkpoint was downloaded, and `manifest_sha256`; for Fortress models the share count, each role's pinned artifact digest (`role_artifacts`), how weight and activation shares are generated, weight and activation bit widths, operation count, expert-pruning fraction |
| `security` | Mode, whether attestation is required, process hardening, prompt armour, memory scrubbing |
| `transport` | Worker names; for remote Fortress models the link and session suites, frame size, cover interval, each worker's route (direct or onion, with hop count) and operator-declared labels, per-request counters, and each worker's signed checkpoint (`worker_checkpoints`) |
| `signer`, `signature` | Public key and signature over the canonical trace. The signature is Ed25519 and ML-DSA-65 together; both must verify |

`transport.counters.fortress_verified_results` counts worker results that passed the client's
Freivalds check during the request. It equals `fortress_role_requests` for a completed request,
because a failed check aborts the request and quarantines the role.

## Model manifest digest

`model.manifest_sha256` is one hash over the repository, revision, source digest, format, layout
digest, architecture, encodings, share count, share generation, weight and activation bit widths,
expert-pruning fraction, and each role's artifact digest. Two traces with the same value ran the
same model build, quantized and split the same way, onto the same role artifacts. `xurcroh verify`
recomputes it and rejects a trace whose model facts do not match.

## Worker checkpoints

Each Fortress role worker started with `--identity-key` holds a long-lived signing key (Ed25519 and
ML-DSA-65 together). The managed deployment creates one per role, keeps it on that role's host
across restarts, and publishes the public key and the digest of the artifacts written for the role
as the worker labels `identity_public` and `artifact_digest`.

Client and worker each keep a running hash of every sealed request and response of their session,
cover calls included. When a request completes, the client asks every role for a checkpoint: the
session, the number of calls, the running hash, the digest of the artifacts the worker loaded, and a
fresh nonce, signed with the worker's identity key. The client rejects the request unless the
signature matches the pinned key, the nonce is its own, the count and hash equal its own, and the
artifact digest equals the pinned one. The checkpoints are stored in the trace, and `xurcroh verify`
checks each signature again and that each worker signed for the artifacts the manifest names.

A verified checkpoint shows that the holder of that identity key answered exactly the calls the
client sent in that session and states which artifacts it loaded. It does not show the worker
computed correctly (the result check does that), where the worker runs, or that the key is held in
protected hardware. No task ID is sent to a worker; the request is an ordinary fixed-size frame and
crosses onion relays unchanged. If no role has a pinned identity, `worker_checkpoints` is `null`
and `xurcroh verify` prints that workers signed nothing.

## Public log

Set `security.trace_log_endpoint` and `security.trace_log_key` to publish each trace hash to a
shared log (`xurcroh.public_log`); `security.trace_log_token_env` names the environment variable
holding the write token. `client.tasks.proof` then returns the public inclusion proof. If the log
is unreachable when a request finishes, the trace is kept and the proof is requested again, and
fails loudly, when it is asked for.

- `xurcroh verify TRACE --proof PROOF --log-key KEY --log-url URL` also asks the log for a
  consistency proof from the proof's tree head to its current one. If the log has dropped or
  rewritten any earlier entry, that check fails.
- `--witness-key KEY` (repeatable) requires a countersignature from that witness on the proof's
  tree head or the current one. A witness (`python -m xurcroh.public_log witness`) keeps the last
  head it saw and countersigns a new one only if it extends it.
- `security.require_logged_manifest` refuses to return output from a model whose manifest digest
  is not in the log. The operator publishes the digest of a build they have checked with
  `python -m xurcroh.public_log publish --kind manifest --digest DIGEST`.
- Release manifests can be logged and checked with the `--log-url` and `--log-key` options of
  `scripts/release.py` and `scripts/verify_release.py`.

What this does and does not give you: an inclusion proof shows the hash was in the log at that tree
size, and a consistency proof shows a later log only appended to it. The log operator can still
show different histories to parties who never compare tree heads; a witness you trust, run by
someone other than the operator, is what narrows that. The project does not operate a public log
or any witness. A deployment has to run them, and until an independent party runs a witness the
log is only as trustworthy as its operator. The client does not check that its own software build
is in the log.

## Hosted gateway

`python -m xurcroh.gateway` serves the client over HTTPS to API-key holders (`xurcroh.gateway`).
It takes a YAML file with `models`, and optionally `cluster` and `security`, a file of
`owner:sha256-of-key` lines, and a trace signing key. Requests use the same routes as the SDK's
HTTP form. Each response carries an `X-Xurcroh-Task-Id` header; the gateway assigns the ID and
ignores one supplied by the requester.

- `GET /v1/tasks/{id}/trace` and `/proof` answer only the API key that made the task. Another key
  gets the same "not found" as for an ID that never existed.
- `GET /v1/trace-key` returns the trace key and log key to pin.
- Every trace is signed with `mode: hosted` and `assurance: operator-asserted`, and
  `xurcroh verify` prints what that means. Removing either field breaks the signature.

What this gives a requester: the trace was not changed after the gateway produced it, it refers to
their prompt and output, and, with a public log, it was not replaced later. What it does not give:
proof that the gateway used the stated model, quantization or settings. The gateway is the trusted
client, so it sees every prompt and output in the clear, and its operator could report anything.
The worker checkpoints in a hosted trace are still signed by the workers, but the requester relies
on the gateway to have checked them against the right identities. Requests are not sealed to an
attested key, because no attested gateway exists. Running the client yourself is the only mode in
which you check the model and settings rather than being told them.

The gateway keeps the API-key digests, the task-to-owner map and the traces in memory for the most
recent `security.trace_history` tasks; a restart loses the map and the traces. It has no rate
limiting, quotas or billing.

## What a trace can and cannot prove

| Claim | Local client | Hosted API |
|---|---|---|
| Workers used the committed model shares | Proven: Freivalds vectors come from the committed weights | Same, if the gateway is honest or attested |
| Model, revision, format, quantization | Proven: the user's client compiled from a digest it checked | Attested gateway only; otherwise "operator-asserted" |
| Settings and seed were applied | Proven: the user's own software sampled | Attested gateway only; otherwise "operator-asserted" |
| Links were encrypted, with which algorithms | Proven: recorded from the live handshake | Same, signed by the gateway |
| Trace was not rewritten later | Proven: hash in the public log | Proven: hash in the public log |

## Limits of the current implementation

- When the client only forwards to a remote worker and holds no copy of the checkpoint (the
  non-Fortress remote engines), it cannot fingerprint the model. The trace then carries
  `model.identity: "unverified: the checkpoint is not held by this client"` and null digests.
- A hosted gateway exists (see "Hosted gateway"). Its traces are always "operator-asserted";
  there is no attested gateway, so the "Attested gateway only" cells in the table above cannot be
  met today.
- The signing key is generated per client process. The signature shows a trace was not changed after
  that client produced it.
- Each trace's hash is appended to a Merkle log (RFC 6962 hashing) and
  `client.tasks.proof(task_id)` returns an inclusion proof against a signed tree head;
  `xurcroh verify --proof proof.json --log-key <hex>` checks it. The logged value is a hash of the
  trace, which already holds only salted commitments, so the log reveals nothing about a request.
  Without further settings this log lives inside the client process with a per-process key, which
  detects a trace changed after logging but is visible to nobody else. See "Public log" for the
  shared one.
- Each remote Fortress route reports under `negotiated` the TLS key-exchange group and the SHA-256
  of the peer's leaf certificate, read from the live connection. With onion routing the peer is the
  entry relay (`link_peer`), so the trace does not show what the later hops negotiated. The same
  entry reports, read from the live session object, whether the role session is established and
  forward-secret (`session_forward_secret`), whether it is bound to the TLS connection
  (`session_bound_to_tls`, true on direct routes and false through relays), and whether the worker
  signed the handshake with its pinned identity (`session_signed_by_worker`).
- Worker labels such as country and CPU vendor are operator declarations and are recorded under
  `declared`. A worker checkpoint (below) identifies a signing key, not a location.
- Image and video traces record size, steps, strength, frames, frame rate, and the effective seed,
  and commit to the prompt, input image, mask, and encoded output. They carry the same sharing facts, routes
  and worker checkpoints as chat traces, but no per-request transport counters, and the source digest of a diffusion pipeline covers its
  `model_index.json` only, not the weight files.
- Traces are held in memory for the most recent `security.trace_history` tasks (default 256) and are
  lost when the client closes. Save a trace if it must outlive the process.
