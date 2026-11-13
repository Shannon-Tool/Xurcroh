# Contributing

Work on a short-lived feature branch or isolated worktree. Read `docs/requirements-matrix.yaml` and `docs/architecture/decisions.md` before changing scope. Fortress supersedes earlier performance-first production assumptions; existing vLLM/SGLang paths are numerical references unless a task explicitly retains them.

For each change:

1. Add a focused failing test that demonstrates the required behavior.
2. Implement the smallest complete change consistent with the public contracts.
3. Run relevant Python, Rust, native, and frontend checks.
4. Update technical documentation, simulations, API references, and requirement evidence affected by the change.
5. Inspect the diff and commit with a scoped Conventional Commit message.

Never commit credentials, private keys, checkpoints, generated shards, raw prompts, sensitive logs, caches, or build artifacts. Do not add mocks or simulations to production inference paths. Keep hardware-only claims explicitly pending until their executable validation has run on the required equipment.
