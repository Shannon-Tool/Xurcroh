# Azure confidential GPU setup

Xurcroh's Azure adapter manages `Standard_NCC40ads_H100_v5` confidential GPU workers, including their NIC, public IP, virtual network, and network security group. Fortress combines fresh CPU/GPU evidence with nonce, workload, artifact, image-measurement, and recipient-key binding. Invalid, stale, or substituted evidence must fail closed.

Azure confidential GPUs were exercised in the frozen validation build. Use that build or reconcile its patches before claiming equivalent acceptance for another source revision.

## 1. Prerequisites

- An Azure subscription with NCC40ads H100 v5 quota in a region where the confidential GPU SKU is available.
- Permission to create resource groups, VMs, NICs, public IPs, virtual networks, and security groups.
- Azure authentication through `DefaultAzureCredential` (workload identity, managed identity, or `az login` for a controlled development client).
- The Xurcroh `cloud` dependencies, installed by `./scripts/bootstrap-python.sh`.
- An SSH public key and explicit trusted source CIDRs. Wildcard ingress is not a supported default.

Set non-secret identifiers in the environment and inject credentials through Azure's credential chain:

```bash
export AZURE_SUBSCRIPTION_ID='00000000-0000-0000-0000-000000000000'
export AZURE_RESOURCE_GROUP='xurcroh-tee'
export XURCROH_ALLOWED_SOURCE_RANGES='203.0.113.0/24'
```

The UUID and CIDR above are documentation examples. Replace them with real non-secret values.

## 2. Inspect the offer

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider azure \
  --offer-id Standard_NCC40ads_H100_v5 \
  --state artifacts/azure/offers-state.json \
  --list-offers
```

Confirm the region, SKU restrictions, quota, GPU model, current price, and confidential-computing terms in Azure. Provider discovery establishes availability, not attestation.

## 3. Run the managed lifecycle smoke test

Create a local, non-secret JSON file outside the repository or under an ignored artifacts directory:

```json
{
  "label": "xurcroh-azure-smoke",
  "ssh_public_key": "ssh-ed25519 AAAA... replace-with-your-public-key",
  "allowed_source_ranges": ["203.0.113.0/24"],
  "disk_gb": 128
}
```

Do not put private keys, access tokens, or service-principal secrets in this file. Then run:

```bash
.venv/bin/python scripts/fortress/tee_provider_smoke.py \
  --provider azure \
  --offer-id Standard_NCC40ads_H100_v5 \
  --values /secure/config/azure-smoke.json \
  --state artifacts/azure/smoke-state.json \
  --output artifacts/azure/smoke-result.json \
  --create
```

The command creates one instance, checks visibility, queries charges, destroys the VM and its worker-specific network resources, and verifies that the instance no longer remains. If execution is interrupted after creation, rerun with `--recover` and the same state path; do not create a duplicate worker. A smoke result explicitly leaves `physical_attestation_validated` false: API lifecycle testing is not attestation testing.

## 4. Deploy attested workers

Use each created VM's address as an SDK SSH worker with security mode `tee`. The validated Fortress build obtains fresh evidence per session and binds it to the handshake, worker key, artifact digest, and expected image measurement. Test stale quotes, wrong evidence, another GPU's evidence, and image substitution before release; each must reject the worker.

Do not release role keys merely because a VM has a confidential-SKU name. Require successful composite CPU/GPU verification for every session, and pin expected images and artifact identities. A TEE reduces host-operator exposure but does not eliminate physical attacks, implementation vulnerabilities, or full-grid collusion.

## 5. Teardown and cost control

Delete all workers through tracked lifecycle state, then independently enumerate instances, disks, NICs, public IPs, virtual networks, and security groups. Reconcile Azure's final bill after delayed usage records arrive. The validated deployment used East US 2 list pricing at the time of testing; current region availability and prices must be checked before every deployment.

References: [Azure confidential computing](https://learn.microsoft.com/azure/confidential-computing/), [Azure confidential VMs](https://learn.microsoft.com/azure/confidential-computing/confidential-vm-overview), and [NCCads H100 v5 VM sizes](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nccads-h100-v5-series).
