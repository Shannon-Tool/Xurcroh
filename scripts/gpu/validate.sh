#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./scripts/bootstrap-python.sh first" >&2
  exit 2
fi
./scripts/build-native.sh
.venv/bin/python -m pytest tests/hardware -q --runxfail
.venv/bin/python -m pytest -q -m 'not hardware'
