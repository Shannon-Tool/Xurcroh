# GPU Topology and Resource Planning

These are analytical provisioning estimates, not reservations, price quotes, or hardware validation results. No provider has been contacted and no paid infrastructure has been used. Select a provider only after checking current availability, checkpoint licenses, independent-operator requirements, network price, and attestation support. Suitable marketplace/provider categories include RunPod, Lambda, Vast.ai, and self-managed GPU hosts; none is endorsed or benchmarked here.

## Memory model

For dense BF16 weights, start with approximately two bytes per parameter per full model, divide by pipeline stages, then add KV cache, activations, CUDA graphs, framework workspaces, communication buffers, and fragmentation headroom. Do not schedule at 100% of device memory. KV cache depends on layer count, KV heads, head dimension, dtype, active sequences, and context length; measure the actual engine configuration. FP8 reduces weight storage but introduces numerical and kernel compatibility requirements, not a free capacity guarantee.

For diffusion/video, include text encoders, transformer weights, VAE/Tiny AutoEncoder, latents, temporal activation workspaces, sequence-parallel buffers, and client decoder memory. The client boundary itself may be too large for a low-end device; offer a trusted client workstation/server profile rather than silently moving boundary secrets to a worker.

## Initial validation topologies

| Workload | Starting topology | Analytical capacity rationale | First acceptance gate |
|---|---|---|---|
| 7–8B dense BF16 | 1 × 24 GB GPU for engine baseline; 2 independent stages for custody tests | 14–16 GB weights plus KV/workspaces | Exact outputs, then failover and transport overhead |
| Llama 70B BF16 | 2–4 × 80 GB GPUs on independent hosts; TP only inside a host group | About 140 GB weights before KV/workspaces | Stage parity, long-context KV, WAN recovery |
| Qwen 235B MoE, FP8 where supported | 4 × 80 GB pipeline stages; EP/TP in high-bandwidth groups | About 235 GB raw FP8 weights plus router/expert workspaces | Routing parity, dropless padding, cache/timing review |
| DeepSeek V3-class MoE | At least 16 × 80 GB GPUs, e.g. two 8-GPU high-bandwidth hosts, adjusted for actual dtype and stages | FP8 weights alone are hundreds of GB; full BF16 is much larger | Checkpoint-specific memory plan, EP/MLA parity, failure isolation |
| Gemma 12B / Qwen 27B VLM | 1 × 80 GB GPU plus a trusted client boundary host | Weights, vision/audio encoders, long multimodal context | Official-config text/image/audio parity and chaff masks |
| Flux / Qwen-Image | 1 × 48–80 GB GPU plus client decoder | Text encoders, diffusion transformer, latent and VAE workspaces | Denoising/reference quality and editing parity |
| Wan 14B-class video | 1 × 80 GB GPU for initial small-context tests; scale sequence parallelism after profiling | Temporal activations and text/VAE components can exceed weight memory | Frame count, temporal quality, sequence-parallel parity |

These configurations are deliberately starting points, not claims that every specified checkpoint fits. Official full checkpoints may require different revisions, quantization, sharding, or more nodes.

## Benchmark budget considerations

Budget separately for GPU-hour burn, CPU/client nodes, checkpoint storage, image volumes, inter-provider egress, and repeated setup failures. Start with short functional runs, then numerical/security tests, then longer throughput and soak runs. Retain raw JSON and deployment revisions. A two-stage acceptance sequence avoids paying for long benchmarks before basic stage parity works. Obtain current provider pricing only during the Project Manager's authorized Stage Two provisioning.

## Outstanding hardware risks

1. Actual vLLM/SGLang/xDiT hook compatibility, kernel numerics, and continuous-batching behavior.
2. End-to-end transformed-stage wiring, pipeline bubbles, communication overlap, and recovery across independent machines.
3. Official-checkpoint output quality for dense, MoE, VLM, image, and video adapters.
4. CUDA wipe ordering, allocator reuse, multi-stream behavior, and hardware regions an epilogue cannot scrub.
5. Attestation trust roots, evidence freshness, workload/key binding, and secondary-device substitution.
6. NAT/WAN relay behavior, congestion, replay/rotation under reconnect, and return-path teardown.
7. MoE cache/TLB/timing/power channels despite constant-load dispatch.
8. Real client memory requirements for boundary weights and image/video decoders.
