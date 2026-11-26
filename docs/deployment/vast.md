# Vast.ai setup

Vast.ai provides independently operated GPU hosts for commodity Fortress grids. Xurcroh can forecast instance cost, require distinct physical machines, provision CUDA workers over SSH, monitor a bounded validation budget, and destroy the instances it created.

Commodity Vast.ai hosts are **not attested confidential workers**. Use the documented share-placement and non-collusion policy, or choose an attested provider for `tee` mode.

## 1. Prepare the account and trusted client

- Add an SSH **public** key to Vast.ai and keep its private key on the trusted client.
- Install the Xurcroh development environment with `./scripts/bootstrap-python.sh`.
- Supply `VAST_API_KEY` through a secret manager or an interactive environment prompt. Never place it in source, a JSON state file, shell history, or a committed configuration.
- Choose enough disk for the pinned checkpoint and compiled artifacts. Estimate network and storage charges in addition to the GPU hourly price.

## 2. Provision a bounded grid

This example creates two distinct 12+ GiB workers for a one-hour smoke test. Increase `--count` to the exact host count required by the selected model and share grid.

```bash
.venv/bin/python scripts/vast/provision.py artifacts/vast/state.json \
  --count 2 \
  --min-gpu-ram-gb 12 \
  --max-hourly-usd 0.75 \
  --disk-gb 80 \
  --planned-hours 1 \
  --ttl-seconds 5400
```

The command refuses a forecast above its configured target, starts a watchdog, and tears down partial resources after failed creation or boot. The default image is a starting point, not a model guarantee; record the actual image, GPU, driver, and CUDA versions.

The provider key is never stored in `artifacts/vast/state.json`. Keep that file and the lock file private because they contain deployment identifiers, addresses, and spending information.

## 3. Deploy and run

Convert each returned `ssh_host`, `ssh_port`, GPU model, and location into a worker entry. Use SSH host-key verification; do not disable it to make provisioning succeed. Follow the [protected multi-host example](../../README.md#protected-multi-host-quickstart), setting `allow_download=True` only for an explicitly pinned model.

```python
with Xurcroh(cluster=workers, model_path=models) as client:
    print(client.preflight())
    response = client.chat.completions.create(
        model="gemma",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)
```

The SDK compiles role artifacts locally, seals them to individual recipients, and removes owned worker resources when the context exits. If cleanup fails, retain the recovery workspace and call `close()` again; do not discard the state before remote destruction is confirmed.

## 4. Monitor and destroy

```bash
.venv/bin/python scripts/vast/status.py artifacts/vast/state.json
.venv/bin/python scripts/vast/destroy_all.py artifacts/vast/state.json
.venv/bin/python scripts/vast/status.py artifacts/vast/state.json
```

`destroy_all.py` checks Vast again for tracked instances. Confirm `all_destroyed: true`, wait for final storage and bandwidth charges, and retain the sanitized ledger. Stopping a pod is not destruction.

The bundled Vast lifecycle enforces its small validation-budget profile (25 USD target, 30 USD maximum). Change that policy only through a separately reviewed deployment configuration; it is not a general production quota system.

References: [Vast.ai API lifecycle](https://docs.vast.ai/api-reference/hello-world), [instance creation](https://docs.vast.ai/api-reference/creating-instances-with-api), [charges](https://docs.vast.ai/api-reference/billing/show-charges), and the repository's [Vast lifecycle details](../vast-validation.md).
