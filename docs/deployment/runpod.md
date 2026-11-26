# RunPod setup

RunPod GPU pods join Xurcroh through the SDK's SSH-managed Fortress deployment. Use GPU pods for model workers; CPU pods are useful for a trusted compiler, gateway, relay, or evidence collector. The current repository does **not** include an automated RunPod provisioning adapter, so create and destroy pods in RunPod or through its official API.

RunPod GPU pods are commodity workers unless a separately verified attestation architecture is supplied. Do not label them as TEE workers.

## 1. Create GPU pods

1. Select a CUDA-compatible PyTorch image and a GPU with enough memory reported by `client.preflight()`.
2. Allocate a persistent volume large enough for the model download, source archive, runtime environment, and recipient-sealed role artifacts.
3. Enable SSH and add only your SSH **public** key. Expose TCP/22 for deployment. Expose UDP/9444 only when the worker accepts direct QUIC; otherwise use the SDK's `datagram_tunnel=true` option.
4. Restrict ingress to the trusted gateway/client addresses where RunPod networking supports it. Do not publish an unauthenticated worker port.
5. Create the required number of pods on separate physical hosts. Record host, external SSH port, username, GPU model, region, and CPU vendor for each pod.

## 2. Configure the SDK

RunPod commonly uses `root` and a provider-assigned external SSH port rather than port 22:

```python
workers = [
    {
        "name": "runpod-a",
        "host": "203.0.113.40",
        "ssh": {"username": "root", "port": 22001, "key_path": "~/.ssh/id_ed25519"},
        "provider": {"country": "us", "cpu_vendor": "amd"},
    },
    {
        "name": "runpod-b",
        "host": "198.51.100.50",
        "ssh": {"username": "root", "port": 22002, "key_path": "~/.ssh/id_ed25519"},
        "provider": {"country": "ca", "cpu_vendor": "intel"},
    },
]
```

Use real addresses and truthful failure-domain metadata. Multiple pods on the same physical host are not independent security boundaries. When direct UDP is unavailable, pass `cluster={"workers": workers, "datagram_tunnel": True}` to `Xurcroh`; the SDK then owns SSH forwarding and relay cleanup.

Deploy using the [protected multi-host quickstart](../../README.md#protected-multi-host-quickstart). Run `client.preflight()` before downloading a large checkpoint and use the client as a context manager to ensure SDK-owned worker processes and artifacts are cleaned up.

## 3. Validate connectivity

Before inference, verify SSH using the exact external port and confirm the NVIDIA runtime inside each pod:

```bash
ssh -p 22001 root@203.0.113.40 'nvidia-smi && python3 --version'
```

Do not bypass host-key checking. If the container uses a different Python interpreter, let the managed bootstrap create the pinned environment rather than installing arbitrary global packages.

## 4. Stop billing safely

Closing the SDK stops Xurcroh workers and removes SDK-owned artifacts; it does not destroy RunPod pods or volumes. After a successful run:

1. Delete the pods through RunPod.
2. Delete volumes that are no longer required.
3. Query RunPod independently to confirm no intended instance or volume remains.
4. Reconcile final compute, storage, and egress charges.

A stopped pod or volume can still incur storage charges. Keep unrelated account resources untouched.

References: [RunPod documentation](https://docs.runpod.io/), [Pods](https://docs.runpod.io/pods/overview), and [SSH access](https://docs.runpod.io/pods/configuration/use-ssh).
