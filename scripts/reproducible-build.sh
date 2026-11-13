#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
first="$work/first"
second="$work/second"
mkdir -p "$first" "$second"
git -C "$root" archive HEAD | tar -x -C "$first"
cp -a "$first/." "$second/"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1700000000}"
export CARGO_INCREMENTAL=0
"$first/scripts/bootstrap-python.sh"
ln -s "$first/.venv" "$second/.venv"
"$first/scripts/build-wheels.sh"
"$second/scripts/build-wheels.sh"
diff -u <(cd "$first" && sha256sum dist/* | sort -k2) <(cd "$second" && sha256sum dist/* | sort -k2)
echo "Reproducible wheel hashes match for $(git -C "$root" rev-parse HEAD)"
