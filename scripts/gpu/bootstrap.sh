#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON="${PYTHON:-python3.12}"
VENV="${XURCROH_CUDA_VENV:-$ROOT/.venv-cuda}"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 5 --retry-all-errors -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.99.1
  export PATH="$HOME/.cargo/bin:$PATH"
fi
"$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir -r requirements-cuda.lock maturin==1.16.0
"$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir --no-deps -e .
VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release --manifest-path crates/transport/Cargo.toml --features python
"$VENV/bin/python" - <<'PY'
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

cpu_vendor = "unknown"
for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("vendor_id"):
        raw_vendor = line.split(":", 1)[1].strip()
        cpu_vendor = {"AuthenticAMD": "AMD", "GenuineIntel": "Intel"}.get(raw_vendor, raw_vendor)
        break

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access a CUDA device")
report = {
    "cpu_vendor": cpu_vendor,
    "torch": torch.__version__,
    "vllm": importlib.metadata.version("vllm"),
    "cuda_runtime": torch.version.cuda,
    "gpus": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ],
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines(),
}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/hardware.json").write_text(json.dumps(report, indent=2) + "\n")
PY
