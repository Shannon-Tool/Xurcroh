# Google Cloud confidential GPU setup

Xurcroh's Google Cloud adapter creates `a3-highgpu-1g` workers with Intel Trust Domain Extensions (TDX), Secure Boot, vTPM, integrity monitoring, and NVIDIA H100 GPUs. It creates a worker-specific firewall rule and removes that rule during teardown.

Google Cloud confidential GPUs were exercised in the frozen validation build. Use that build or reconcile its patches before claiming equivalent acceptance for another source revision.

## 1. Prerequisites

- A Google Cloud project with Compute Engine and A3 HighGPU quota in a TDX-capable zone.
- Permission to create instances and firewall rules, view instance state, and query billing.
- Application Default Credentials through workload identity, a managed runtime, or `gcloud auth application-default login` for a controlled development client.
- The Xurcroh `cloud` dependencies, installed by `./scripts/bootstrap-python.sh`.
- An SSH public key and explicit source CIDRs.

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
export XURCROH_ALLOWED_SOURCE_RANGES='203.0.113.0/24'
# Optional: fully qualified project.dataset.table for final cost queries.
export GOOGLE_BILLING_EXPORT_TABLE='your-project.billing_export.gcp_billing_export_v1'
```

Use a configured BigQuery detailed cost export for parameterized charge queries. The export table is optional for creation but necessary for adapter-based billing evidence. Credentials stay in Google Application Default Credentials; never copy a service-account private key into Xurcroh configuration.

## 2. Inspect the offer

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider gcp \
  --offer-id a3-highgpu-1g \
  --state artifacts/gcp/offers-state.json \
  --list-offers
```

The adapter enumerates zones where `a3-highgpu-1g` is available. Independently confirm TDX and NVIDIA confidential-computing support, capacity, Spot availability, maintenance behavior, and current price for the selected zone. The adapter requests Spot provisioning and `TERMINATE` maintenance behavior; interruptions must be handled by the deployment and recovery policy.

## 3. Run the managed lifecycle smoke test

Create a non-secret JSON file outside the repository or under an ignored artifacts directory:

```json
{
  "label": "xurcroh-gcp-smoke",
  "ssh_public_key": "ssh-ed25519 AAAA... replace-with-your-public-key",
  "allowed_source_ranges": ["203.0.113.0/24"],
  "disk_gb": 128
}
```

No private key or OAuth token belongs in this file. Run:

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider gcp \
  --offer-id a3-highgpu-1g \
  --values /secure/config/gcp-smoke.json \
  --state artifacts/gcp/smoke-state.json \
  --output artifacts/gcp/smoke-result.json \
  --create
```

The command checks creation and visibility, queries available charge data, destroys the instance and its worker-specific firewall, and verifies remote absence. After an interrupted lifecycle, use `--recover` with the original state file. `physical_attestation_validated` remains false in a lifecycle smoke report; a provider API response is not hardware evidence.

## 4. Deploy attested workers

Configure each VM as an SDK SSH worker with security mode `tee`. The validated Fortress build verifies fresh Intel TDX and NVIDIA evidence against current collateral and binds it to the challenge nonce, handshake, workload/artifact identity, recipient key, and expected image measurement.

Reject stale evidence, wrong quotes, second-GPU substitution, and changed image measurements before admitting any worker. Do not silently fall back to commodity mode when attestation fails. Shielded VM settings and a TDX instance type are necessary configuration inputs, not proof by themselves.

## 5. Teardown and cost control

Destroy all workers from tracked state, then independently enumerate instances, persistent disks, external addresses, and worker-specific firewall rules. Reconcile BigQuery and final invoice data after delayed records settle. Include the confidential-computing surcharge and network egress in deployment estimates. Validation-time prices are historical measurements, not current quotes.

References: [Google Cloud Confidential Computing](https://cloud.google.com/confidential-computing), [Confidential VM](https://cloud.google.com/confidential-computing/confidential-vm/docs/confidential-vm-overview), and [A3 machine series](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-machine-series).
