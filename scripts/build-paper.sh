#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1795737600}"
export FORCE_SOURCE_DATE=1
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
for command in latexmk pdflatex biber; do
  command -v "$command" >/dev/null || {
    echo "Missing $command; see paper/README.md" >&2
    exit 1
  }
done
"$PYTHON" "$ROOT/paper/scripts/generate_figures.py"
cd "$ROOT/paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -jobname=xurcroh-fortress main.tex
