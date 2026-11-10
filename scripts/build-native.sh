#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler nvcc is required to build Xurcroh native memory controls" >&2
  exit 2
fi
ARCH="${CUDA_ARCH:-sm_90}"
OUT="$ROOT/native/build"
mkdir -p "$OUT"
nvcc -O3 -arch="$ARCH" -Xcompiler -fPIC -shared \
  "$ROOT/native/cuda/xurcroh_burn_alloc.cu" -o "$OUT/xurcroh_burn_alloc.so"
nvcc -O3 -arch="$ARCH" -Xcompiler -fPIC -shared \
  "$ROOT/native/cuda/otp_mask_kernels.cu" -o "$OUT/otp_mask_kernels.so"
printf '%s\n' "$OUT"
