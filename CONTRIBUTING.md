# Contributing

Work on a short-lived feature branch or isolated worktree. Read `AGENTS.md`, the three specifications, and `docs/requirements/matrix.yaml` before changing scope.

For each change:

1. Add a focused failing test that demonstrates the required behavior.
2. Implement the smallest complete change consistent with the public contracts.
3. Run relevant Python, Rust, native, and frontend checks.
4. Update technical documentation, simulations, API references, and requirement evidence affected by the change.
5. Inspect the diff and commit with a scoped Conventional Commit message.

Never commit credentials, private keys, checkpoints, generated shards, raw prompts, sensitive logs, caches, or build artifacts. Do not add mocks or simulations to production inference paths. Keep hardware-only claims explicitly pending until their executable validation has run on the required equipment.
