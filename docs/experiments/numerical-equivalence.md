# Numerical Equivalence Experiment

**Evidence class:** verified by local CPU experiment, not a GPU benchmark.

Reproduce with:

```bash
.venv/bin/python experiments/numerics.py --iterations 50
.venv/bin/pytest tests/algebra tests/experiments/test_numerics.py -q
```

The experiment uses a deterministic 32→64 linear projection, independently seeded orthogonal bases, and the same FP32 weights and activations for every precision. `max_absolute_error` compares transformed and reference computations after applying the same reduced-precision rounding. `max_reference_error` compares the reduced-precision recovered result with the original unrounded FP32 result; this second value is the meaningful model-error measurement.

On Python 3.12.3 with Torch 2.15.0+cpu:

| Output precision | Maximum equivalence error | Maximum error against original FP32 |
|---|---:|---:|
| FP32 | 9.54e-6 | 9.54e-6 |
| BF16 | 0.0 | 3.97e-2 |
| FP16 | 2.24e-5 | 5.00e-3 |
| FP8 E4M3 | 0.0 | 5.80e-1 |

Zero equivalence error after rounding does not mean zero quality loss. FP8's substantially larger reference error demonstrates why reduced-precision equivalence must be assessed separately from algebraic exactness. These are synthetic single-layer measurements, not a model-wide quality bound.

The same run measured a 32×4096 keyed FWHT at approximately 4.68 ms per transform over 50 iterations on this CPU. This is a local observation, not a target or cross-hardware performance claim. Machine-readable values are in `numerical-results.json`.

The experiment initially exposed a double basis multiplication in its own comparison code. Regression tests now check both the rotated-space comparison and the original-reference error, preventing a broken experiment from silently reporting meaningless results.

## Llama RoPE and distributed boundary check

`tests/models/test_dense_pipeline.py::test_compiled_pipeline_matches_huggingface_llama_logits` compares a real two-stage compiled Llama boundary against `AutoModelForCausalLM`, rather than only against Xurcroh's own reference functions. It exposed that the initial implementation used adjacent-pair RoPE while the checkpoint uses half-split rotary layout. The compiler now selects the checkpoint-compatible layout and the test passes with FP64 tolerance `atol=3e-8, rtol=1e-6`; the largest observed absolute difference before tolerance adjustment was approximately `1.49e-8`. This is numerical equivalence within tolerance, not bit-for-bit equivalence or a claim about every RoPE variant.

`tests/integration/test_stage_sdk.py` executes the same compiled model through two actual HTTP worker processes with authenticated encrypted tensors, client-owned embedding/LM-head sampling, synchronous and streaming responses, and cancellation. Development HTTP is only the local test transport; strict deployments require TLS stage endpoints. Physical QUIC GPU-stage execution remains hardware-pending.
