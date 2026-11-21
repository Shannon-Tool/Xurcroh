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
| `model` | Name, revision and whether it is pinned, source format, digest of the source checkpoint files, layout digest, architecture, encodings; for Fortress models the share count, how weight and activation shares are generated, weight and activation bit widths, operation count, expert-pruning fraction |
| `security` | Mode, whether attestation is required, process hardening, prompt armour, memory scrubbing |
| `transport` | Worker names; for remote Fortress models the link and session suites, frame size, cover interval, each worker's route (direct or onion, with hop count) and operator-declared labels, and per-request counters |
| `signer`, `signature` | Ed25519 public key and signature over the canonical trace |

`transport.counters.fortress_verified_results` counts worker results that passed the client's
Freivalds check during the request. It equals `fortress_role_requests` for a completed request,
because a failed check aborts the request and quarantines the role.

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
- Only the local mode exists. There is no hosted gateway, so no trace is "operator-asserted" yet.
- The signing key is generated per client process. The signature shows a trace was not changed after
  that client produced it.
- Each trace's hash is appended to a Merkle log (`xurcroh.merkle_log`, RFC 6962 hashing) and
  `client.tasks.proof(task_id)` returns an inclusion proof against a signed tree head;
  `xurcroh verify --proof proof.json --log-key <hex>` checks it. The logged value is a hash of the
  trace, which already holds only salted commitments, so the log reveals nothing about a request.
  This log lives inside the client process and its key is generated per process. It detects a trace
  changed after logging, but it is not public and has no independent witness, so the last row of
  the table above is not yet delivered. A hosted log service and consistency checking between tree
  heads are still to be built on the same proofs.
- Each remote Fortress route reports under `negotiated` the TLS key-exchange group and the SHA-256
  of the peer's leaf certificate, read from the live connection. With onion routing the peer is the
  entry relay (`link_peer`), so the trace does not show what the later hops negotiated. The session
  suite string is the configured one.
- Workers do not yet sign anything; worker labels such as country and CPU vendor are operator
  declarations and are recorded under `declared`.
- Image and video traces record size, steps, strength, frames, frame rate, and the effective seed,
  and commit to the prompt, input image, mask, and encoded output. They carry no per-request
  transport counters or sharing facts yet, and the source digest of a diffusion pipeline covers its
  `model_index.json` only, not the weight files.
- Traces are held in memory for the most recent `security.trace_history` tasks (default 256) and are
  lost when the client closes. Save a trace if it must outlive the process.
- The signature is Ed25519, which is not post-quantum.
