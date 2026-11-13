#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="${WHEEL_DIR:-$ROOT/dist}"
home="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
cargo_home="${CARGO_HOME:-$home/.cargo}"
export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }--remap-path-prefix=${cargo_home}=/cargo --remap-path-prefix=${ROOT}=/workspace"
if [[ -n "${CARGO_BIN:-}" ]]; then
  export PATH="$(dirname "$CARGO_BIN"):$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust is required; install the toolchain from rust-toolchain.toml or set CARGO_BIN" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found at $PYTHON; run ./scripts/bootstrap-python.sh" >&2
  exit 2
fi
mkdir -p "$OUT"
"$PYTHON" -m pip wheel --no-deps --no-build-isolation . --wheel-dir "$OUT"
"$PYTHON" -m maturin build --release \
  --manifest-path crates/transport/Cargo.toml --features python --out "$OUT"
"$PYTHON" "$ROOT/scripts/normalize-wheel-sbom.py" \
  --project-root "$ROOT" "$OUT"/*.whl
printf '%s\n' "$OUT"
