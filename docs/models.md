# Supported Model Matrix

Xurcroh accepts checkpoint-driven adapters with immutable revisions and explicit licence review. **Hardware validated** means the official checkpoint completed the Fortress ladder on physical GPUs across independent hosts with exact equality to the quantized reference, measured source-model quality, adversarial checks, and signed evidence. Results below apply to frozen build `3b48e2a + patches-01…23` (`7f31c2e0`).

## Production-validated text and multimodal models

| Model | Pinned revision | Engine and validated workflows | Hardware result |
|---|---|---|---|
| Gemma 4 E2B | `google/gemma-4-E2B-it` @ `4adf15748de9e1a0097b6274cee9c15836c41c37` | Fortress; distributed text, VLM boundaries, streaming, recovery | L0 and attested D2-a passed; quantized error 0; Apache-2.0 |
| Gemma 4 E4B | `google/gemma-4-E4B-it` @ `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | Fortress; distributed text and confidential execution | L1a and attested D2-b passed; quantized error 0; Apache-2.0 |
| Gemma 4 12B | `google/gemma-4-12B-it` @ `43c9d014b231474fa7212dbb48a762566c2b255d` | Fortress; distributed text | L1b passed; quantized error 0; Apache-2.0 |
| Gemma 4 26B A4B | `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | Fortress MoE; client routers, decoy calls, padded dispatch | L2 passed; quantized error 0; Apache-2.0 |
| Gemma 4 31B | `google/gemma-4-31B-it` @ `842da3794eaa0b77d5f08bae87a17459d91ff475` | Fortress; distributed text | L3a passed; quantized error 0; Apache-2.0 |
| Qwen 3.8 27B | `Qwen/Qwen3.8-27B` @ `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | Fortress VLM; text plus image/audio conditioning | L3b passed; quantized error 0; Apache-2.0 |
| Mistral Medium 3.5 128B | `mistralai/Mistral-Medium-3.5-128B` @ `ff62230e1c990bdf05bbf6efa1c73fefe094ebf4` | Fortress dense; FP8 `F8_E4M3` decoding | L4 passed; quantized error 0; Mistral Medium Research License |
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash` @ `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | Fortress MoE; 288 experts × 45 layers, FP8 | L5 passed; quantized error 0; MIT |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` @ `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | Fortress MoE; packed `I32`, measured REAP at kept fraction 0.472 | L6 passed; quantized error 0; 4.1% perplexity delta vs unpruned; Kimi K2 Community License |

## Production-validated image and video models

| Model | Pinned revision | Validated workflows | Hardware result |
|---|---|---|---|
| FLUX.2 klein 4B | `black-forest-labs/FLUX.2-klein-4B` @ `e7b7dc27f91deacad38e78976d1f2b499d76a294` | Text-to-image, non-square/odd-sized and chunked-temporal shapes | M1a passed; exact latents; PSNR 44.2 dB; Apache-2.0 |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` @ `f332072aa78be7aecdf3ee76d5c247082da564a6` | Few-step text-to-image | M1b passed; exact latents; PSNR 43.7 dB; Apache-2.0 |
| Qwen-Image 2.1 | `Qwen/Qwen-Image-2.1` @ `d26bb61231c349cf6b7896fa83353113880e1ba3` | Text-to-image, image-to-image, masked editing | M2a passed; exact latents; PSNR 42.8 dB; Qwen research licence |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` @ `e8af80c9dd820ab07ee4171d024f03e5c4b3024b` | Text-to-image, image-to-image, masked editing | M2b passed; exact latents; PSNR 42.1 dB; gated non-commercial licence |
| Wan 2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff7315c768468a5333511427288870b2e9635` | Text/image-to-video, deterministic MP4 | M3a passed; exact latents; VMAF 96.4; Apache-2.0 |
| LTX-2.5 | `Lightricks/LTX-2.5-Diffusers` @ `4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3` | Video generation | M3b passed; exact latents; VMAF 95.8; gated LTX community licence |
| Wan 2.2 T2V-A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` @ `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | Text-to-video, dual denoiser | M4a passed; exact latents; VMAF 95.2; Apache-2.0 |
| Wan 2.2 I2V-A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` @ `596658fd9ca6b7b71d5057529bbf319ecbc61d74` | Image-to-video | M4b passed; exact latents; VMAF 94.9; Apache-2.0 |
| MiniMax H3 | `MiniMaxAI/MiniMax-H3` @ `12253298adb81e36757eabebb81d2ba1c0cc70cc` | Full video generation, recurrent state, joint audio | M4c passed; exact latents and audio checks; VMAF 94.1; MiniMax H3 community licence |

## Quantized container support

The Fortress compiler accepts Safetensors, GGUF v1–v3, EXL2/GPTQ, and EXL3 on the trusted client. GGUF covers all 37 current type codes, shards, metadata arrays, embedded tokenizers, and both endian modes. EXL2 covers mixed-width groups and activation ordering; EXL3 covers integer/fractional trellis rates, all three codebooks, and sign/Hadamard restoration. FP8 `F8_E4M3` and packed-integer `I32` decoders were differentially verified against publisher loaders.

GGUF graph semantics require an explicit entry in `python/xurcroh/architectures/gguf_families.py` backed by an official-export differential comparison. Validated families are `llama`, `qwen2`, `qwen3`, `qwen3moe`, `gemma3`, and `gemma4`; unsupported graph layouts fail closed. EXL2/EXL3 retain the source `config.json` and execute through the checkpoint's Transformers model type when its graph is supported. Format smoke and official ExLlama CUDA differential validation passed in G2.

## Execution and licence notes

Public SDK registration selects Fortress automatically and supports synchronous, streaming, and asynchronous chat, plus unified multimodal methods. MoE padding is dropless with zero dummy combine weights. VLM chaff is attention-masked. Strict remote DiT execution uses compiled equivariant weights without basis or permutation parameters in worker requests. No production method returns simulated text or media.

Model support is limited to the validated families and compatible checkpoint graphs above; future architectures are not automatically claimed. Every deployment must use a pinned revision and comply with the model's licence, including gated and non-commercial terms. See `docs/xurcroh-validation-report.md` for fleet configurations, quality measurements, and acceptance conditions.
