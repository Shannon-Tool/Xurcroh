#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler nvcc is required to build Xurcroh native memory controls" >&2
  exit 2
fi
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)"
  ARCH="sm_${CAPABILITY/./}"
else
  ARCH="$CUDA_ARCH"
fi
OUT="$ROOT/native/build"
mkdir -p "$OUT"
nvcc -O3 -arch="$ARCH" -Xcompiler -fPIC -shared \
  "$ROOT/native/cuda/xurcroh_burn_alloc.cu" \
  "$ROOT/native/cuda/otp_mask_kernels.cu" \
  -o "$OUT/xurcroh_burn_alloc.so"
printf '%s\n' "$OUT"
