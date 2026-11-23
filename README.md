# Project Xurcroh

Xurcroh is a distributed inference platform for dense and Mixture-of-Experts language models, vision-language models, image diffusion, and video generation. The Fortress architecture combines exact integer secret sharing, client-owned nonlinear computation, encrypted shards, a Rust QUIC data plane, memory lifecycle controls, and hardware attestation for the older stage engines (Fortress role workers do not yet run attested). Existing algebraic/vLLM paths are transition and numerical-reference components, not the Fortress production engine.

Every completed request produces a signed task trace that can be checked with `xurcroh verify` or in a browser on the manual's Verify page; see `docs/task-trace.md` and `docs/claims.md`.

> **Security status:** designed for defence in depth; claims under evaluation. Fortress implementation and production evidence are incomplete. Orthogonal transformations are computational obfuscation, not homomorphic encryption or a cryptographic non-reversibility proof.

## Public Python API

```python
from Xurcroh import Xurcroh

client = Xurcroh(
    cluster=[{"name": "local", "endpoint": "local://cpu"}],
    model_path={"qwen-7b": "/models/qwen-7b"},
)

response = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
client.close()
```

Synchronous, asynchronous, streaming, multimodal, advanced cluster, and security configuration are documented in `docs/sdk.md`. GGUF, EXL2/GPTQ, and EXL3 checkpoints use the same Fortress API with automatic format detection and explicit semantic adapters; unknown graph semantics fail closed. A requested model must resolve to a real execution adapter; Xurcroh never returns generated-looking canned responses. Deployment and repeatable JSON benchmarks are documented in `docs/deployment.md` and `docs/benchmarking.md`.

## Protected multi-host quickstart

From a committed source checkout, SSH-reachable CUDA workers can be provisioned automatically with per-operation placement. The trusted client downloads the pinned checkpoint, compiles encrypted role artifacts, starts the workers, and tears them down when the context exits. This Gemma topology uses 20 workers with alternating geographic and CPU-vendor metadata:

```python
from Xurcroh import Xurcroh

workers = [
    {
        "name": f"gpu-{index}",
        "host": f"10.0.0.{10 + index}",
        "ssh": {"username": "ubuntu", "key_path": "~/.ssh/id_ed25519"},
        "provider": {
            "country": "us" if index % 2 else "de",
            "cpu_vendor": "amd" if index % 2 else "intel",
        },
    }
    for index in range(20)
]

with Xurcroh(
    cluster=workers,
    model_path={
        "gemma": {
            "path": "google/gemma-4-E2B-it",
            "revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
            "engine": "fortress",
            "allow_download": True,
        }
    },
) as client:
    response = client.chat.completions.create(
        model="gemma",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=128,
    )
    print(response.choices[0].message.content)
```

A runnable CLI version is available at `examples/managed_gemma.py`; supply each worker with repeated `--host` arguments and use `--datagram-tunnel` when direct UDP is unavailable. Use `--shares N` for per-operation `N × N` grids and provide exactly `5N²` hosts; the default is `--shares 2` with 20 hosts. Set `--countries` and `--cpu-vendors` to one value each or one per host to describe the actual infrastructure. Use a trusted gateway with sufficient host memory and disk for checkpoint compilation. `client.preflight()` reports measured checkpoint size and resource issues before inference. This is the implemented deployment interface, not a claim that the official Gemma checkpoint has completed physical multi-host validation.

## Development

```bash
./scripts/bootstrap-python.sh
.venv/bin/pytest
```

The interactive research manual source is under `apps/web-manual/` and is independent of the production runtime. GPU topology estimates and outstanding hardware risks are in `docs/resources.md`.

## Security

Report vulnerabilities privately as described in `SECURITY.md`. Never commit credentials, private keys, checkpoints, raw prompts, generated shards, or sensitive runtime logs.
