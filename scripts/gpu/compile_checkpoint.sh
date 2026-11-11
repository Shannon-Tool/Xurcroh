#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT OUTPUT_DIR STAGE_COUNT" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CHECKPOINT="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
STAGES="$3"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
[[ "$STAGES" =~ ^[1-9][0-9]*$ ]] || { echo "STAGE_COUNT must be positive" >&2; exit 2; }
[[ -f "$CHECKPOINT/config.json" ]] || { echo "checkpoint config.json is missing" >&2; exit 2; }
mkdir -p "$OUTPUT/identities" "$OUTPUT/client"
chmod 700 "$OUTPUT/identities" "$OUTPUT/client"
mapfile -t RECIPIENT_ARGS < <("$PYTHON" - "$OUTPUT" "$STAGES" <<'PY'
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

output = Path(sys.argv[1])
for index in range(int(sys.argv[2])):
    path = output / "identities" / f"stage-{index}.key"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    key = X25519PrivateKey.generate()
    private = key.private_bytes_raw()
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(private)
    print(f"--stage-recipient={index}={public.hex()}")
PY
)
if [[ ! -f "$OUTPUT/client/boundary-key.hex" ]]; then
  "$PYTHON" - "$OUTPUT/client/boundary-key.hex" <<'PY'
import os
import secrets
import sys

descriptor = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    handle.write(secrets.token_hex(32) + "\n")
PY
fi
IFS= read -r BOUNDARY_KEY < "$OUTPUT/client/boundary-key.hex"
export XURCROH_CLIENT_BOUNDARY_KEY="$BOUNDARY_KEY"
unset BOUNDARY_KEY
"$PYTHON" -m xurcroh.cli compile \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT" --stages "$STAGES" \
  --compute-dtype "${XURCROH_COMPUTE_DTYPE:-f32}" \
  --shard-dtype "${XURCROH_SHARD_DTYPE:-bf16}" \
  "${RECIPIENT_ARGS[@]}"
"$PYTHON" - "$CHECKPOINT" "$OUTPUT/worker-checkpoint" <<'PY'
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(mode=0o755, exist_ok=True)
for name in (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
):
    if (source / name).is_file():
        shutil.copyfile(source / name, destination / name)
if list(destination.glob("*.safetensors")):
    raise SystemExit("plaintext weights must never enter worker-checkpoint")
PY
for ((index=0; index<STAGES; index++)); do
  mkdir -p "$OUTPUT/stage-$index"
  cp "$OUTPUT/manifest.json" "$OUTPUT/stage-$index.xurcroh" "$OUTPUT/stage-$index.key.json" \
    "$OUTPUT/stage-$index/"
done
printf '%s\n' "$OUTPUT"
