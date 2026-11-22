"""Write the fixtures that keep the browser trace verifier in step with the Python one.

Every file is produced by the production signing and logging code, with fresh keys, so
the TypeScript port in ``apps/web-manual/lib/verify`` is tested against real output
rather than against itself. Run from the repository root:

    python scripts/generate_verify_fixtures.py apps/web-manual/tests/fixtures/verify.json
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from xurcroh.fortress.checkpoint import CHECKPOINT_CONTEXT, CHECKPOINT_FORMAT, _canonical
from xurcroh.public_log import COSIGNATURE_CONTEXT, PublicLog, _witnessed
from xurcroh.receipts import (
    canonical,
    canonical_media,
    canonical_messages,
    commit,
    model_manifest_digest,
    sign_trace,
    trace_digest,
    verify_trace,
    verify_trace_proof,
)
from xurcroh.signatures import HybridSigningKey, key_fingerprint

MESSAGES = [
    {"role": "system", "content": "Répondez brièvement. 日本語も可。"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": 'What is "2 + 2"?\n\ttab \\ slash   😀'},  # noqa: RUF001
            {"type": "input_audio", "input_audio": {"data": "AAAA"}},
        ],
    },
]
OUTPUT = "Four — quatre — 四 \u007f"
MEDIA = {"prompt": "a red kite over a grey sea", "image": None, "mask": None}


def _checkpoint(worker: str, key: HybridSigningKey, artifacts: str) -> dict[str, object]:
    body = {
        "format": CHECKPOINT_FORMAT,
        "session": "5f" * 16,
        "calls": 42,
        "transcript": "ab" * 32,
        "artifacts": artifacts,
        "weight_index": 1,
        "nonce": "cd" * 32,
        "identity": key_fingerprint(key.public_key),
    }
    return {
        "worker": worker,
        "body": body,
        "identity_public": key.public_key.hex(),
        "signature": key.sign(CHECKPOINT_CONTEXT, _canonical(body)).hex(),
        "artifacts_pinned": True,
    }


def build() -> dict[str, Any]:
    signer = HybridSigningKey.generate()
    workers = {f"role-{index}": HybridSigningKey.generate() for index in range(2)}
    role_artifacts = {name: f"{index + 1:02x}" * 32 for index, name in enumerate(workers)}
    model: dict[str, object] = {
        "name": "tiny-llama",
        "repository": "example/tiny-llama",
        "revision": "0123abcd",
        "source_format": "safetensors",
        "source_digest": "11" * 32,
        "layout_digest": "22" * 32,
        "architecture": "llama",
        "encodings": ["int8"],
        "share_count": 2,
        "weight_shares": "uniform",
        "weight_bits": 8,
        "activation_bits": 16,
        "reap_keep_fraction": 0.1,
        "role_artifacts": role_artifacts,
    }
    model["manifest_sha256"] = model_manifest_digest(model)
    document: dict[str, Any] = {
        "task_id": "task-é-1",
        "mode": "local",
        "created_at": 1795305600.123457,
        "completed_at": 1795305601.5,
        "client": {"version": "0.1.0"},
        "model": model,
        "request": {
            "prompt": commit(canonical_messages(MESSAGES)),
            "settings": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": 7},
            "effective": {"temperature": 1e-07, "scale": 1e22, "small": 5e-324},
        },
        "result": {"output": commit(OUTPUT.encode()), "tokens": 3, "ratio": -0.0},
        "security": {"mode": "fortress", "note": "naïve ≠ verified"},
        "transport": {
            "worker_checkpoints": [
                _checkpoint(name, key, role_artifacts[name]) for name, key in workers.items()
            ]
        },
    }
    trace = sign_trace(document, signer)

    # Traces a dishonest signer could produce: the outer signature is valid, an inner
    # fact is not. Only a verifier that re-checks the inner parts rejects them.
    def resigned(change: Any) -> dict[str, Any]:
        altered = copy.deepcopy(document)
        change(altered)
        return sign_trace(altered, signer)

    def forge_checkpoint(altered: dict[str, Any]) -> None:
        altered["transport"]["worker_checkpoints"][0]["body"]["calls"] = 1

    def swap_artifacts(altered: dict[str, Any]) -> None:
        stranger = HybridSigningKey.generate()
        altered["transport"]["worker_checkpoints"][1] = _checkpoint("role-1", stranger, "99" * 32)

    def edit_model(altered: dict[str, Any]) -> None:
        altered["model"]["revision"] = "ffff0000"

    media_trace = sign_trace(
        {
            "task_id": "media-1",
            "mode": "hosted",
            "assurance": "operator-asserted",
            "request": {"prompt": commit(canonical_media(MEDIA))},
            "result": {},
            "transport": {"worker_checkpoints": None},
        },
        signer,
    )
    verify_trace(trace, public_key=signer.public_key, messages=MESSAGES, output=OUTPUT)
    verify_trace(media_trace, public_key=signer.public_key, media_input=MEDIA)

    log_key = HybridSigningKey.generate()
    witness = HybridSigningKey.generate()
    with tempfile.TemporaryDirectory() as directory:
        log = PublicLog(directory, log_key)
        for index in range(5):
            log.append("manifest", f"{index + 16:02x}" * 32)
        log.append("trace", trace_digest(trace).hex())
        log.append("release", "ee" * 32)
        head = log.tree_head()
        log.add_cosignature(
            head["tree_size"],
            {
                "witness_key": witness.public_key.hex(),
                "signature": witness.sign(COSIGNATURE_CONTEXT, _witnessed(head)).hex(),
            },
        )
        position = log.lookup("trace", trace_digest(trace).hex())
        assert position is not None
        proof = log.proof(position)
    verify_trace_proof(trace, proof, log_key=log_key.public_key)
    return {
        "trace": trace,
        # The exact bytes a user would have on disk; number spelling must survive parsing.
        "trace_text": json.dumps(trace, ensure_ascii=False, indent=2),
        "trace_canonical_sha256": trace_digest(trace).hex(),
        "body_canonical": canonical(
            {name: value for name, value in trace.items() if name != "signature"}
        ).decode(),
        "media_trace": media_trace,
        "forged_checkpoint_trace": resigned(forge_checkpoint),
        "wrong_artifacts_trace": resigned(swap_artifacts),
        "wrong_manifest_trace": resigned(edit_model),
        "messages": MESSAGES,
        "output": OUTPUT,
        "media": MEDIA,
        "signer_public": signer.public_key.hex(),
        "other_public": HybridSigningKey.generate().public_key.hex(),
        "proof": proof,
        "log_key": log_key.public_key.hex(),
        "witness_key": witness.public_key.hex(),
    }


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n")
