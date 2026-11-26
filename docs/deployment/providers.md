# Cloud deployment

Xurcroh runs one inference grid across commodity GPU hosts, confidential virtual machines, or a combination of providers. The SDK discovers worker capacity, places independent shares across failure domains, compiles recipient-sealed model artifacts, and opens authenticated QUIC sessions. Choose the provider below, then use the same `from Xurcroh import Xurcroh` API for every platform.

| Platform | Execution boundary | Setup guide | Xurcroh provisioning |
|---|---|---|---|
| Vast.ai | Commodity NVIDIA GPUs on independent hosts | [Vast.ai](vast.md) | Automated offer selection, lifecycle guardrails, SSH deployment, and destruction |
| RunPod | GPU pods or trusted CPU infrastructure | [RunPod](runpod.md) | SDK-managed workers over SSH; create and destroy pods in RunPod |
| Microsoft Azure | Confidential GPU VMs with hardware evidence | [Azure confidential GPUs](azure-tee.md) | Provider API lifecycle plus SSH-managed Fortress workers |
| Google Cloud | A3 HighGPU with Intel TDX and NVIDIA attestation | [Google Cloud confidential GPUs](gcp-tee.md) | Provider API lifecycle plus SSH-managed Fortress workers |

For another confidential-hardware option, see the Alibaba Cloud adapter in [Deployment and Operations](../deployment.md#confidential-cloud-provider-adapters). CPU-only TEE adapters are for trusted control-plane or client components, not confidential GPU workers.

## Before you deploy

1. Use a trusted client or gateway with enough disk and memory to hold and compile the checkpoint. Worker capacity alone is not sufficient.
2. Pin every model revision and review its license. Start with `client.preflight()` to check checkpoint size, host count, capacity, and topology before inference.
3. Use a clean, committed source checkout. SDK-managed deployment sends the committed source to workers rather than silently uploading uncommitted edits.
4. Give every worker a distinct SSH host. Supply accurate country and CPU-vendor metadata so complementary shares do not land in the same declared failure domain.
5. Allow SSH/TCP 22 and Fortress QUIC/UDP 9444 only from explicit trusted source CIDRs. Use SSH agent forwarding or a local key file; never put private keys or provider tokens in model configuration.
6. Keep tensor and expert groups inside a high-bandwidth domain. Place independent Fortress shares across separate hosts and operators.
7. Set security mode deliberately. Commodity workers require the documented non-collusion assumption; `tee` mode must fail closed when evidence is unavailable or invalid.
8. Destroy resources after use. A stopped instance can continue to incur storage charges, and destroying only the VM can leave billable disks or network resources.

## Minimal worker configuration

Provider-created hosts become ordinary SDK worker entries:

```python
workers = [
    {
        "name": "gpu-a",
        "host": "203.0.113.10",
        "ssh": {"username": "root", "port": 22, "key_path": "~/.ssh/id_ed25519"},
        "provider": {"country": "us", "cpu_vendor": "amd"},
    },
    {
        "name": "gpu-b",
        "host": "198.51.100.20",
        "ssh": {"username": "ubuntu", "port": 22, "key_path": "~/.ssh/id_ed25519"},
        "provider": {"country": "de", "cpu_vendor": "intel"},
    },
]
```

A `shares=2` Gemma grid normally needs 20 independently addressable hosts (`5 × 2²`); add hosts until `client.preflight()` accepts the placement. Do not substitute multiple ports on one physical machine for independent workers in a security-sensitive deployment.

See the [complete SDK example](../../README.md#protected-multi-host-quickstart), [model matrix](../models.md), and [hardware validation report](../xurcroh-validation-report.md). Production acceptance is scoped to frozen build `3b48e2a + patches-01…23`, content hash `7f31c2e0`, and the conditions in that report; installing the provider dependencies alone does not reproduce that validated build.
