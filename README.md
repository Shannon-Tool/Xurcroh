# Project Xurcroh

Xurcroh is a production-oriented, zero-trust distributed inference platform for dense and Mixture-of-Experts language models, vision-language models, image diffusion, and video generation. The implementation combines client-owned boundaries, algebraic model transformations, encrypted model shards, a Rust QUIC data plane, memory lifecycle controls, and optional hardware attestation.

> **Validation status:** pre-GPU engineering. Physical multi-GPU execution, production security acceptance, and hardware performance are not yet validated. Orthogonal transformations are computational obfuscation, not homomorphic encryption or a cryptographic non-reversibility proof.

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

The requirement evidence lives under `docs/`. The interactive research manual source is under `apps/web-manual/` and is independent of the production runtime. GPU topology estimates and outstanding hardware risks are in `docs/resources.md`. `docs/handoff.md` distinguishes locally verified implementation from remaining pre-GPU engineering work.

## Security

Report vulnerabilities privately as described in `SECURITY.md`. Never commit credentials, private keys, checkpoints, raw prompts, generated shards, or sensitive runtime logs.
