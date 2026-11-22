# Blind Controller and Worker Lifecycle

## Worker identity and health

Workers register with a fresh, canonical `Registration` signed with Ed25519 and ML-DSA-65 together (`xurcroh.signatures`). The controller verifies the public-key signature, freshness, and endpoint stability; it never receives the private key. Periodic signed re-registration replaces the former unsigned heartbeat endpoint and keeps workers healthy; stale workers expire. Capabilities include model names, modalities, engine, group, GPU model/memory, and admission capacity. Impersonation, stale registration, and endpoint reassignment fail closed.

## Circuits and relay

`TopologyPlanner` selects eligible healthy workers, enforces distinct pipeline stages, and requires TP/EP degrees to fit one named high-bandwidth host group. Circuit records contain stage assignments only. Client return secrets are not controller state.

`POST /v1/relay/{circuit_id}` forwards an opaque byte body to the selected stage with no parsing or prompt logging. Stage workers execute that payload and return an independently encrypted result; the generic worker echo route has been removed. `DenseStageEngine` creates one circuit per generation, relays through it when `controller_endpoint` is configured, and deletes it in a generator-finally path on completion, cancellation, or failure. Request-scoped KV caches are removed through `DELETE /v1/relay/{circuit_id}/stages/{request_id}` before circuit teardown. Integration tests use an unroutable direct endpoint to prove controller execution and verify zero remaining circuits and stage-cache entries. Development HTTP is available only for local tests. Production workers require HTTPS; QUIC stage transport uses the Rust pinned-certificate implementation. Onion-style return keys appear only in the innermost encrypted layer. Physical NAT behavior is not claimed verified by loopback tests.

## Actual remote inference

`RemoteOpenAIEngine` calls a real Xurcroh worker's `/v1/chat/completions`; responses are computed by the same lazy-loaded model engine used by the public SDK. Non-stream and SSE streaming responses include usage and execution metadata. The client also accepts the earlier NDJSON stream format for compatibility. Alternates are tried before output on connection failures. Once a stream emits content, retry stops to avoid duplicate tokens. HTTPS or explicit development HTTP is required; strict mode rejects plaintext HTTP and full-model remote checkpoints. Strict remote execution requires a compiled stage manifest and stage workers.

`python3 -m xurcroh.worker` starts a real worker with repeated `--model name=/path` options, optional controller registration, and required production TLS certificate/key arguments. SSH provisioning discovers GPU metadata with `nvidia-smi` and starts workers only when `provision_workers=true` and a controller endpoint is explicitly configured. SSH private keys/passwords are not embedded in commands or logs.

## Admission, recovery, and metrics

Per-model bounded semaphores limit concurrency and produce typed admission timeouts. Request IDs, cancellation, and cleanup are independent across users. Managed streams release admission and registration even when closed before their first token. Remote cancellation stops retry; timeout and connection failures have structured exceptions.

Metrics contain only counters and durations: started, completed, failed, timed-out, cancelled, and total duration. They exclude prompts, tokens, media, credentials, and circuit secrets. Use `client.metrics()` or the worker `/metrics` endpoint for operational integration; asynchronous clients provide `await client.metrics()`.
