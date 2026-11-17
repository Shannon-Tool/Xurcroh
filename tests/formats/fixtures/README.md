# GGUF test fixtures

GGUF binaries and model checkpoints are intentionally not committed. `tests/formats/gguf_utils.py` generates tiny synthetic containers for each supported encoding during tests; expected quantized values come from the pinned `gguf` decoder where available. `scripts/formats/generate_gguf_fixtures.py <directory>` writes representative F32, Q4_0, Q1_0, Q2_0, and F8_E4M3 fixtures for manual inspection. These tensors contain no model or user data.
