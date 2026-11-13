#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON="${PYTHON:-$ROOT/.venv-cuda/bin/python}"
./scripts/build-native.sh
"$PYTHON" -m pytest tests/hardware -q --runxfail
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "compute-sanitizer is required for the forensic gate" >&2
  exit 2
fi
compute-sanitizer --tool initcheck "$PYTHON" -m pytest \
  tests/hardware/test_fortress_cuda.py tests/hardware/test_cuda_forensics.py -q --runxfail
