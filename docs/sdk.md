# Python SDK Reference

## Installation and lifecycle

Xurcroh requires Python 3.12 or 3.13. The distribution and import namespace are both `Xurcroh`. Use a context manager so active requests are cancelled and model resources are released:

```python
from Xurcroh import Xurcroh

with Xurcroh(
    cluster={"endpoint": "local://cpu"},
    model_path={"qwen-7b": "/models/qwen-7b"},
) as client:
    response = client.chat.completions.create(
        model="qwen-7b",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7,
        max_tokens=1024,
    )
    print(response.choices[0].message.content)
```

`AsyncXurcroh` provides the same resources with `async with`, `await ...create(...)`, and `async for` streaming. Its constructor does not provision or download synchronously; shared initialization is offloaded to a worker thread on `async with` or the first async operation, with concurrent callers coalesced behind one lock. Both clients load model weights lazily. Calls on a closed client raise `ClientClosedError`, and closing an uninitialized async client does not create one.

## Models

`model_path` accepts one local path or a mapping of public names to paths/configuration dictionaries. Advanced definitions support `path`, `revision`, `engine`, `modality`, `allow_download`, `manifest_path`, `boundary_key_env`, `protect_latents`, and `latent_security`. Downloads are disabled by default. Unknown names raise `ModelNotFoundError` with the available registry; no model name is silently remapped. For Fortress checkpoints, `modality="auto"` inspects `model_index.json` and selects image or video execution without requiring redundant configuration. Gemma 4 and Qwen 3.5 default to text-only decoder loading; select `modality="vlm"` for the official processor and full multimodal model. In that mode, vision/audio towers remain on the trusted client while language-decoder linear operations use the Fortress grid. When `manifest_path` names a compiled `.xurcroh` manifest, remote workers execute encrypted dense stages while the trusted client retains tokenization, embedding, final normalization, LM-head sampling, and decoded text. The stage count must match `pipeline_parallel` and the selected worker count. The independent client-boundary key is read indirectly from `boundary_key_env` (default `XURCROH_CLIENT_BOUNDARY_KEY`) and is never a response or log field. `engine="vllm"` or `engine="sglang"` selects optimized stages; strict `auto` requires optimized execution, while `engine="reference"` is development-only. `protect_latents=true` enables protected media execution. `latent_security="equivariant"` requires a compiled DiT manifest and sends no basis/permutation secrets to workers; `latent_security="transport_only"` is development-only worker-side obfuscation. Remote sessions use ephemeral X25519 agreement, directional AEAD tensor envelopes, client-side prompt conditioning/noise, and client-side decode; workers return transformed latents, not decoded images or video. Source-image editing uses validated `strength` and client VAE encoding, so source media is not forwarded to the worker. Unsupported pipelines fail explicitly instead of silently running unprotected.

```python
client = Xurcroh(
    cluster={"endpoint": "local://cpu"},
    model_path={
        "qwen-7b": "/models/qwen-7b",
        "llama-70b": {
            "path": "/models/llama-70b",
            "engine": "vllm",
            "modality": "text",
        },
    },
)
```

## Checkpoint preflight

`client.preflight()` reads checkpoint metadata and safetensors storage without loading model weights or starting inference. Its return values use `CheckpointPreflight`, exported directly from `Xurcroh`; async media methods likewise return concrete image/video response types rather than `Any`. Pass a registered model name to check only that model. Each report includes the checkpoint layout digest, measured checkpoint bytes, conservative client and per-role memory estimates, and actionable issues. Fortress requests run this check before engine creation. With `allow_download=true`, a hub model is resolved to an immutable 40-character commit before download. Existing local checkpoints instead use their measured content layout. Automatic provisioning requires known memory from discovery or `gpu.memory_gb`/`resources.max_gpu_memory_gb`; known insufficient capacity is always rejected. Existing Fortress grids are also checked for a complete unique role matrix, valid recipient keys, QUIC endpoints, and readable client TLS files before checkpoint compilation.

## Cluster configuration

A cluster can be an endpoint string, one worker dictionary, a worker list, or an advanced mapping with `workers`, `pipeline_parallel`, `tensor_parallel`, `expert_parallel`, `security`, `provision_workers`, and `datagram_tunnel`. Set `datagram_tunnel=true` on managed clusters when direct UDP is unavailable; each model receives an SSH-forwarded local QUIC relay on its own role port. Relay forwarding requires SSH keys or an agent rather than password authentication. An empty configuration creates one explicit `local://cpu` worker. For Fortress models, SSH-configured hosts are automatically provisioned unless `provision_workers=false` is explicit. The SDK discovers GPU capacity, compiles per-role encrypted artifacts from a committed source revision, bootstraps the pinned CUDA environment, starts mutually authenticated QUIC role workers, and owns their teardown. Automatic provisioning currently expects one complete role grid per model: four workers for the default two-share configuration. Set the model's `fortress_shares` to `N` for an `N × N` grid with exactly `N²` unique worker roles; values below two are rejected. Non-Fortress models are rejected by automatic provisioning rather than started as plaintext workers. Remote endpoints without SSH require initialized Xurcroh workers.

Worker fields include `name`, `host`, `endpoint`, `ssh`, `gpu`, `labels`, `group`, `provider`, `network_interface`, and `resources`. SSH supports `username`, `port`, `key_path`, `password`, `password_env`, `known_hosts_path`, and strict host-key checking. `SecretStr` redaction prevents passwords from appearing in representations. Operators should prefer SSH keys, agents, or indirectly supplied environment credentials. Secrets are never valid log fields.

For separately managed Fortress endpoints, the optional `labels.operations` value is a comma-separated list of compiled operation names assigned to that worker; the default `*` preserves the full-grid behavior. Every assigned operation must have a complete `N × N` role grid. `apply_placement(workers, placement)` converts a `PlacementMap` into `operations` and per-operation `operation_roles` labels, and providers retain a consistent weight-share role across their operations. Operations with identical worker sets share authenticated connections, while disjoint sets execute on independent grid runtimes. Automatic SSH provisioning currently emits `*` and does not yet generate or consume a `PlacementMap`.

Parallelism is validated before execution; pipeline degree cannot exceed worker count. Strict security rejects plaintext worker weights and unauthenticated transport. `tee` mode requires attestation. Development-only downgrades must be selected explicitly and are not production security configurations.

## Chat completions

`client.chat.completions.create` supports `model`, `messages`, `temperature`, `top_p`, `top_k`, `max_tokens`, `stream`, `timeout`, `request_id`, `seed`, and `stop`. Temperature zero selects greedy decoding. Local, dense-stage, and Fortress engines accept generation-config EOS lists as well as single stop-token IDs. Responses use OpenAI-style `choices`, `message`, `usage`, and streaming `delta` objects. `execution` contains request ID, actual engine, worker IDs, duration, and security mode, never prompt contents.

```python
stream = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    request_id="request-42",
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Call `client.cancel("request-42")` to cancel active work. Cancellation and timeout raise `RequestCancelledError` and `RequestTimeoutError`. Local Transformers, VLM, and Fortress inference-lock acquisition use the request deadline through one shared helper, so waiting for another request cannot silently exceed the configured timeout. Concurrent requests use independent request state and actual model execution. Closing a stream or client releases its request registration; abandoned streams cannot deadlock shutdown.

## Images and video

Chat messages accept plain text or typed content parts: `{"type": "text", "text": "..."}`, `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`, and `{"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}}`. Image inputs support base64 data URLs and trusted local `file://` paths; remote URL fetching is disabled. WAV input must be signed 16-bit PCM. Text-only models reject media parts before execution.

`client.images.generate`, `client.images.edit`, and `client.videos.generate` use validated request schemas and typed responses. A model whose detected modality does not support the requested operation raises `CapabilityError`. A supported model without a running diffusion/video endpoint raises `ClusterUnavailableError`; these methods never return synthetic media. `Images.edit` requires `image`. Video supports image-conditioned generation through the same field. Local and remote VLM/image/video engines use the same public methods; remote media endpoints are `/v1/images/generations`, `/v1/images/edits`, and `/v1/videos/generations`.

## Operational metrics

`client.metrics()` returns aggregate started, completed, failed, timed-out, cancelled, and total-duration counters. Fortress grids additionally report request/byte totals, reconnections, cover traffic, and cover-failure counts without operation names, prompts, or endpoint identifiers. `await AsyncXurcroh.metrics()` provides the same snapshot asynchronously. Workers expose `/metrics`. These values exclude prompts, tokens, media, credentials, and circuit secrets; restrict metric endpoint access because aggregate activity can still reveal operational behavior.

## Errors

All stable exceptions derive from `XurcrohError`: configuration, model lookup/loading, capability, cluster, protocol, attestation, shard, request, cancellation, timeout, compatibility, and closed-client failures. Invalid requests fail before model execution where possible. The SDK does not expose stack-trace-only Pydantic failures as its configuration contract.

The command-line interface provides real local inference:

```bash
xurcroh chat --model tiny --model-path /models/tiny --prompt 'Hello!' --max-tokens 32
```


## Fortress execution

Select `engine="fortress"` to compile every attention and feed-forward projection into quantized additive share grids. Embeddings, normalization gains, routers, the LM head, and sampling stay on the client. The same public chat API works for local execution and four-role remote grids:

```python
from Xurcroh import Xurcroh

client = Xurcroh(
    cluster=cluster_config,
    model_path={
        "qwen": {
            "path": "/models/qwen",
            "engine": "fortress",
            "fortress_seed": 731,
        }
    },
)
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0,
)
```

Each remote worker needs `activation_index`, `weight_index`, and `recipient_public` labels plus a QUIC endpoint. The client identity uses `client_certificate_path`, `client_key_path`, and `ca_path`; optional labels set `server_name`, `frame_bytes`, `cover_interval`, and `seeded_activations`. Seeded activation transport is enabled by default for grids and provides computational hiding; set `seeded_activations=false` for raw information-theoretic activation shares. Every worker and the client must use the same compiled operation set. `fortress_seed` makes bases and weight shares reproducible for a distributed deployment and must remain a client-held secret; omit it for fresh local compilation. Freivalds challenges remain fresh and are not derived from the deployment seed. Do not compile workers and client independently without a shared seed or an exported shared package.

Fortress is not production-accepted merely because these API tests pass. Model quality, multi-host hardware execution, side-channel resistance, and external security review remain separate gates.
