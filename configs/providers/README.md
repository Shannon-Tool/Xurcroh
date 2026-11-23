# TEE provider lifecycle validation

These commands provision real billable infrastructure. Run them only with explicit authorization and provider credentials supplied through the provider's normal secret mechanism. The implementation pass did not run them. They validate provider APIs and teardown, not attestation, model inference, or production acceptance.

Install the full environment first:

```bash
./scripts/bootstrap-python.sh
.venv/bin/maturin develop --manifest-path crates/transport/Cargo.toml --features python
```

Create a non-secret JSON file containing the provider-specific creation fields listed in `docs/deployment.md`. SSH public keys are permitted; private keys, passwords, API keys, session keys, and attestation secrets are not. Restrict inbound CIDRs to the trusted client/controller networks.

List an actual offer without creating infrastructure:

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider azure --offer-id Standard_NCC40ads_H100_v5 \
  --state /secure/azure-smoke-state.json --list-offers
```

Create one instance, query its charge, destroy it, and verify that no instance remains:

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider azure --offer-id Standard_NCC40ads_H100_v5 \
  --values /secure/azure-create.json \
  --state /secure/azure-smoke-state.json \
  --output /secure/evidence/azure-lifecycle.json --create
```

Use the equivalent `gcp`/`a3-highgpu-1g` or `alibaba`/`ecs.gn8v-tee.4xlarge` selection for a confidential-GPU path. OCI, AWS, and IBM validate CPU-TEE lifecycle only and cannot satisfy a GPU gate. An offer reported by the provider can still be unavailable at creation time because of capacity or quota; record that as a failed run, not a pass.

If the process exits after creation, recover the recorded instance instead of creating another:

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider azure --offer-id Standard_NCC40ads_H100_v5 \
  --state /secure/azure-smoke-state.json \
  --output /secure/evidence/azure-lifecycle-recovery.json --recover
```

The state file is mode `0600` and records only provider, offer, and instance identity. It never stores creation credentials. `status: destroyed` plus `remaining: false` is required for lifecycle success. Charge APIs can settle late; rerun the provider's charge query before finalizing the invoice ledger. If the provider disallows post-delete charge queries, the lifecycle report remains `destroyed` and marks `charge_query_after_destroy_available: false` instead of misclassifying successful teardown as a provider failure. `physical_attestation_validated` remains `false` in this lifecycle report and must be established separately using the provider-specific evidence procedure.
