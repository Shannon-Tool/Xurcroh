# Project Xurcroh

Xurcroh is a distributed inference platform for dense and Mixture-of-Experts language models, vision-language models, image diffusion, and video generation. The Fortress architecture combines exact integer secret sharing, client-owned nonlinear computation, encrypted shards, a Rust QUIC data plane, memory lifecycle controls, and optional hardware attestation. Existing algebraic/vLLM paths are transition and numerical-reference components, not the Fortress production engine.

> **Security status:** designed for defence in depth; claims under evaluation. Fortress implementation and independent evidence are incomplete. Orthogonal transformations are computational obfuscation, not homomorphic encryption or a cryptographic non-reversibility proof. See `docs/roadmap.md` and `docs/validation-report.md`.

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

Synchronous, asynchronous, streaming, multimodal, advanced cluster, and security configuration are documented in `docs/sdk.md`. A requested model must resolve to a real execution adapter; Xurcroh never returns generated-looking canned responses. Deployment and repeatable JSON benchmarks are documented in `docs/deployment.md` and `docs/benchmarking.md`.

## Development

```bash
./scripts/bootstrap-python.sh
.venv/bin/pytest
```

The Fortress roadmap and requirement evidence live under `docs/roadmap.md` and `docs/requirements-matrix.yaml`. The interactive research manual source is under `apps/web-manual/` and is independent of the production runtime. GPU topology estimates and outstanding hardware risks are in `docs/resources.md`. Actual validation evidence and remaining acceptance work are in `docs/validation-report.md`.

## Security

Report vulnerabilities privately as described in `SECURITY.md`. Never commit credentials, private keys, checkpoints, raw prompts, generated shards, or sensitive runtime logs.
