#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON="${PYTHON:-python3.12}"
FORTRESS_ONLY="${XURCROH_FORTRESS_ONLY:-0}"
export FORTRESS_ONLY
VENV="${XURCROH_CUDA_VENV:-$ROOT/.venv-cuda}"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
export CARGO_HOME="${XURCROH_CARGO_HOME:-$ROOT/.cargo}"
export RUSTUP_HOME="${XURCROH_RUSTUP_HOME:-$ROOT/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
  for attempt in 1 2 3 4 5; do
    if curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 5 --retry-all-errors -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.99.1; then
      break
    fi
    if [[ "$attempt" == "5" ]]; then
      echo "Rust toolchain installation failed after five attempts" >&2
      exit 1
    fi
    sleep "$((attempt * 5))"
  done
fi
export RUSTUP_TOOLCHAIN="1.99.1"
[[ -x "$CARGO_HOME/bin/cargo" ]] || { echo "pinned Cargo executable is missing" >&2; exit 1; }
IMAGE_PACKAGES=0
if "$PYTHON" - <<'PY'
import importlib.metadata
import os
expected = {"torch": "2.9.1"}
if os.environ.get("FORTRESS_ONLY") != "1":
    expected.update({"torchvision": "0.24.1", "vllm": "0.15.0", "transformers": "4.57.6"})
for name, version in expected.items():
    installed = importlib.metadata.version(name).split("+", 1)[0]
    if installed != version:
        raise SystemExit(1)
PY
then
  IMAGE_PACKAGES=1
fi
VENV_ARGS=()
if [[ "$IMAGE_PACKAGES" == "1" ]]; then
  VENV_ARGS=(--system-site-packages)
fi
"$PYTHON" -m venv "${VENV_ARGS[@]}" "$VENV"
if [[ "$FORTRESS_ONLY" == "1" ]]; then
  "$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir maturin==1.16.0
  if [[ "$IMAGE_PACKAGES" != "1" ]]; then
    "$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir -r requirements-fortress-cuda.lock
  fi
elif [[ "$IMAGE_PACKAGES" == "1" ]]; then
  "$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir pytest==9.2.0 pytest-asyncio==1.4.1.post1 maturin==1.16.0
else
  "$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir -r requirements-cuda.lock maturin==1.16.0
fi
"$VENV/bin/pip" install --retries 10 --timeout 120 --no-cache-dir --no-deps -e .
VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release --manifest-path crates/transport/Cargo.toml --features python
"$ROOT/scripts/build-native.sh"
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
try:
    vllm_version = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    vllm_version = None
report = {
    "cpu_vendor": cpu_vendor,
    "torch": torch.__version__,
    "vllm": vllm_version,
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
