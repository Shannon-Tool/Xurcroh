# Repeatable Benchmarks

Benchmark commands execute the real algorithm or public SDK request. They never generate synthetic inference responses. Reports are versioned JSON and exclude prompt text, output text, image bytes, video bytes, credentials, and keys. Output paths under `benchmark-results/` are ignored by Git.

## Commands

After `./scripts/bootstrap-python.sh` and the Rust binding build:

```bash
.venv/bin/maturin develop --manifest-path crates/transport/Cargo.toml --features python

.venv/bin/xurcroh-benchmark fwht \
  --dimension 4096 --iterations 20 --output benchmark-results/fwht.json
.venv/bin/xurcroh-benchmark transport \
  --payload-bytes 65536 --iterations 100 --output benchmark-results/transport.json
.venv/bin/xurcroh-benchmark memory \
  --dimension 4096 --rows 128 --iterations 20 --device cpu \
  --output benchmark-results/memory-cpu.json

.venv/bin/xurcroh-benchmark chat \
  --config benchmarks/cluster.example.yaml --model qwen-7b \
  --prompt 'Explain transformers.' --iterations 5 --max-tokens 64 \
  --output benchmark-results/chat.json
.venv/bin/xurcroh-benchmark image \
  --config image-cluster.yaml --model flux --prompt 'A city at sunset' \
  --size 1024x1024 --steps 30 --reference reference.png \
  --output benchmark-results/image.json
.venv/bin/xurcroh-benchmark video \
  --config video-cluster.yaml --model wan --prompt 'Ocean waves' \
  --size 832x480 --frames 81 --fps 24 --steps 40 --reference reference.mp4 \
  --output benchmark-results/video.json
```

The `chat` command streams actual responses and measures time to first chunk, per-chunk intervals, total latency, prompt/completion usage, and tokens/second. This is an end-to-end measurement, not a separately instrumented engine prefill timer. Use the hardware-stage benchmark plan for isolated prefill/decode and single-node versus multi-node comparisons.

`fwht` checks the result against direct matrix multiplication and inversion before reporting throughput. `transport` verifies Rust AEAD and tensor-frame round trips and compares them with a memory-copy baseline; it does not claim network throughput. `memory` records actual process RSS, Python allocation high-water, and selected allocator counters. `--device cuda` requires a CUDA device and synchronization; CPU counters do not prove physical GPU erasure.

Image and video commands measure actual PNG/MP4 responses. Reference comparisons report MAE, MSE, and PSNR over decoded RGB pixels. These metrics are not substitutes for perceptual model-quality evaluation, task accuracy, or human review. Video reference and generated frame dimensions must match.

## Report interpretation

Every report has `schema_version: 1`, a workload kind, timestamp, parameters, and environment. `environment.hardware_validated` is `false` for the supplied local harness. Performance varies with host, affinity, thermal state, checkpoint, dtype, batch, network, and framework revision. Run sufficient warmup and repetitions on the deployment hardware and retain machine configuration and raw JSON. The 40–50 ms/token architectural target is not a measured result. Measured production-validation fleet, memory, bandwidth, media-quality, and L6 operation-latency results are published in `docs/xurcroh-validation-report.md`; those figures describe the validation environment, not a service-level guarantee.

For hardware validation, capture at least:

- single-worker and actual pipeline-stage baselines with identical checkpoint and sampling;
- prefill and steady-state inter-token latency distributions, not just means;
- aggregate throughput under concurrent independent users;
- plaintext-development versus authenticated strict transport overhead;
- host/GPU allocated and reserved memory, KV-cache cleanup, and fragmentation;
- image/video latency, encoded size, decoded frame count, reference metrics, and task-quality evaluation.

No benchmark result files are committed as evidence of unavailable hardware.
