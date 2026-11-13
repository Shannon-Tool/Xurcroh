from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_fortress_cuda_image_is_minimal_and_uses_role_worker() -> None:
    dockerfile = (ROOT / "deploy/fortress-cuda.Dockerfile").read_text()
    lock = (ROOT / "requirements-fortress-cuda.lock").read_text().splitlines()
    assert "xurcroh.fortress.worker_cli" in dockerfile
    assert "nvidia/cuda:12.8.1-devel-ubuntu24.04" in dockerfile
    assert "XURCROH_NATIVE_LIBRARY" in dockerfile
    assert not any(line.startswith("vllm==") for line in lock)
    assert "torch==2.9.1" in lock
