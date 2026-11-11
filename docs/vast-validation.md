# Vast.ai validation lifecycle

Paid provisioning is permitted only after the exact corrective branch SHA has passed local gates and GitHub Actions. The automated policy targets **$25 total** and never permits a configured target above the **$30 absolute authorization**. A forecast includes instance compute, local disk, and a configurable bandwidth reserve. Stopping an instance is not teardown: Vast states that storage billing begins at creation and continues while stopped. Destruction is mandatory.

The implementation follows the official [REST lifecycle](https://docs.vast.ai/api-reference/hello-world), [instance creation fields](https://docs.vast.ai/api-reference/creating-instances-with-api), and [per-instance charges API](https://docs.vast.ai/api-reference/billing/show-charges). Endpoint responses are reduced to non-sensitive fields before they are returned or persisted. Provider keys and instance API keys are never written to the state file.

## Preflight

1. Confirm `git status --porcelain` is empty and record `git rev-parse HEAD`.
2. Confirm the GitHub Actions run for that exact SHA is green.
3. Compile the real checkpoint on the trusted client and verify the two worker bundles.
4. Register an SSH public key with Vast and load the private key into the local SSH agent. Host-key checking remains enabled.
5. Supply the provider credential only through the process environment or an approved secret manager. Do not place it in configuration files, shell history, logs, or source.

```bash
export VAST_API_KEY  # set indirectly by the secret manager
.venv/bin/python scripts/vast/provision.py artifacts/vast/state.json \
  --count 1 --min-gpu-ram-gb 12 --max-hourly-usd 0.75 \
  --disk-gb 40 --planned-hours 1 --ttl-seconds 5400
```

The default image is `vllm/vllm-openai:v0.15.0`; record the resolved provider image and actual hardware before execution. The initial model target is `Qwen/Qwen3-0.6B`, with its exact reviewed checkpoint revision recorded separately. Large checkpoints are intentionally excluded from the first smoke test.

For two independent workers, first prepare `artifacts/qwen3-compiled` and then use `--count 2`. Offers must have distinct `machine_id` values. The script forecasts all nodes before creating the first one and destroys partial resources if creation or boot fails. `artifacts/vast/two-node.json` is the input to `scripts/gpu/deploy_two_node.sh`.

## Monitoring and watchdog

Provisioning starts a detached watchdog that reconciles charges and estimated spend every 30 seconds. It destroys tracked instances at the $25 threshold, after repeated provider failures, or when the 90-minute default TTL expires. Its PID is stored in the state file; its output contains no provider credential. The watchdog destroys only instance IDs created by this Xurcroh state file and does not delete unrelated account resources.

```bash
.venv/bin/python scripts/vast/status.py artifacts/vast/state.json
```

The status command also triggers teardown when the threshold is reached. Charges can arrive after usage; reported charges and conservative elapsed-time estimates are both retained, and the larger value is used for budget enforcement. Bandwidth estimates are not a substitute for final provider charges. A final ledger remains marked `charges_finalized: false` until a later charge refresh establishes the final invoice total.

## Mandatory teardown

```bash
.venv/bin/python scripts/vast/destroy_all.py artifacts/vast/state.json
# After billing settles, reconcile again without creating resources:
.venv/bin/python scripts/vast/status.py artifacts/vast/state.json
```

Destruction retries failed deletes, treats an already absent instance as idempotently deleted, lists account instances afterward, and fails if any tracked ID remains. `artifacts/vast/destruction.json` records tracked IDs, remaining IDs, errors, estimated/reported cost, and verification time. `all_destroyed: true` applies only to tracked Xurcroh instances and must be supported by the remote list response. Unrelated resources are explicitly left untouched.

Never claim a zero balance from a stopped instance, a local PID exit, or a successful DELETE response alone. Keep the JSON state, charge history, deployment report, and destruction report together as validation evidence.
