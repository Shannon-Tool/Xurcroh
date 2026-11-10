export type Evidence = "derived" | "published" | "local" | "hardware" | "estimate" | "hypothesis" | "limitation";

export interface Section {
  slug: string;
  title: string;
  summary: string;
  file: string;
  evidence: Evidence;
}

export const sections: Section[] = [
  { slug: "01-introduction", title: "Introduction", summary: "Objectives, scope, terminology, and the client/controller/worker boundary.", file: "01-introduction.mdx", evidence: "derived" },
  { slug: "02-research-foundations", title: "Research Foundations", summary: "Equivariance, private inference, QUIC, and confidential computing literature.", file: "02-research-foundations.mdx", evidence: "published" },
  { slug: "03-threat-model", title: "Threat Model", summary: "Trust boundaries, collusion, physical attacks, and residual leakage.", file: "03-threat-model.mdx", evidence: "limitation" },
  { slug: "04-mathematics", title: "Mathematics", summary: "Conjugation, normalization, attention, recurrence, proofs, and counterexamples.", file: "04-mathematics.mdx", evidence: "derived" },
  { slug: "05-system-architecture", title: "System Architecture", summary: "The unified production runtime and its cross-language contracts.", file: "05-system-architecture.mdx", evidence: "derived" },
  { slug: "06-model-sharding", title: "Model Sharding", summary: "Checkpoint compilation, encrypted shard format, manifests, and strict loading.", file: "06-model-sharding.mdx", evidence: "local" },
  { slug: "07-inference-runtime", title: "Inference Runtime", summary: "vLLM, SGLang, xDiT, batching, and pipeline execution.", file: "07-inference-runtime.mdx", evidence: "local" },
  { slug: "08-secure-transport", title: "Secure Transport", summary: "Rust QUIC, AEAD, replay windows, rotation, and onion metadata.", file: "08-secure-transport.mdx", evidence: "local" },
  { slug: "09-memory-security", title: "Memory Security", summary: "GPU/CPU lifecycle, KV caches, stream ordering, and zeroization limits.", file: "09-memory-security.mdx", evidence: "limitation" },
  { slug: "10-dense-transformers", title: "Dense Transformers", summary: "GQA attention, feed-forward networks, stage transitions, and decoding.", file: "10-dense-transformers.mdx", evidence: "local" },
  { slug: "11-moe-architecture", title: "MoE Architecture", summary: "Expert permutation, MLA, constant-load dispatch, and side channels.", file: "11-moe-architecture.mdx", evidence: "local" },
  { slug: "12-vision-language", title: "Vision-Language", summary: "Gemma and Qwen boundaries, audio, IMRoPE, DeltaNet, and DeepStack.", file: "12-vision-language.mdx", evidence: "local" },
  { slug: "13-image-diffusion", title: "Image Diffusion", summary: "Flux and Qwen-Image, AdaLN, joint attention, and local decoding.", file: "13-image-diffusion.mdx", evidence: "local" },
  { slug: "14-video-generation", title: "Video Generation", summary: "Wan and MiniMax, recurrent states, sparse attention, and decoding.", file: "14-video-generation.mdx", evidence: "local" },
  { slug: "15-interactive-laboratory", title: "Interactive Laboratory", summary: "Executable conjugation, FWHT, pipeline, onion, MoE, diffusion, and memory labs.", file: "15-interactive-laboratory.mdx", evidence: "local" },
  { slug: "16-performance", title: "Performance", summary: "Complexity, analytical budgets, benchmarks, and measurement policy.", file: "16-performance.mdx", evidence: "estimate" },
  { slug: "17-security-evaluation", title: "Security Evaluation", summary: "Attacks, defenses, local evidence, and unsupported assurances.", file: "17-security-evaluation.mdx", evidence: "local" },
  { slug: "18-deployment", title: "Deployment", summary: "Reproducible setup, worker provisioning, networking, and validation.", file: "18-deployment.mdx", evidence: "local" },
  { slug: "19-developer-reference", title: "Developer Reference", summary: "Public SDK, source map, protocols, errors, and API contracts.", file: "19-developer-reference.mdx", evidence: "local" },
  { slug: "20-troubleshooting", title: "Troubleshooting", summary: "Failure signatures, debugging, and known constraints.", file: "20-troubleshooting.mdx", evidence: "limitation" },
  { slug: "21-research-references", title: "Research References", summary: "Primary papers, standards, and official framework sources.", file: "21-research-references.mdx", evidence: "published" },
  { slug: "22-experimental-reports", title: "Experimental Reports", summary: "Reproducible numerical and integration experiments with actual outcomes.", file: "22-experimental-reports.mdx", evidence: "local" },
];

export const glossary = [
  { term: "Boundary", definition: "Client-owned embedding, projection, output-head, decoding, and secret lifecycle." },
  { term: "Conjugation", definition: "Offline weight transformation that preserves computation in a changed basis under stated constraints." },
  { term: "Contragredient pair", definition: "Two transformations whose composition cancels even when they are not orthogonal." },
  { term: "Chaff", definition: "Synthetic padding excluded by exact attention masks rather than relied upon as semantic privacy." },
  { term: "Constant-load dispatch", definition: "Dropless MoE dispatch with equal per-expert work buffers and zero-weight dummy tokens." },
  { term: "Evidence label", definition: "A claim's status: derivation, publication, local experiment, hardware benchmark, estimate, hypothesis, or limitation." },
];
