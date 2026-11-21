"""Local theory-mode execution of every media ladder rung.

Each rung of ``configs/ladders/media-ladder.yaml`` runs with a public miniature
pipeline of the rung's own component classes through four loopback role workers
and the real ``verify_ladder`` path. The fixtures are fetched explicitly (see
``scripts/fortress/fetch_tiny_pipelines.py``); without them the tests skip.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.test_ladder import manifest_mapping

from xurcroh.config import ModelDefinition
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import sdk_configuration
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
from xurcroh.fortress.media_components import compile_media_checkpoint
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker import FortressRoleWorker

FIXTURES = os.environ.get("XURCROH_TINY_PIPELINES", "")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not FIXTURES,
        reason="set XURCROH_TINY_PIPELINES to the directory written by "
        "scripts/fortress/fetch_tiny_pipelines.py",
    ),
]

LADDER = Path(__file__).resolve().parents[2] / "configs" / "ladders" / "media-ladder.yaml"
RUNGS: list[dict[str, Any]] = yaml.safe_load(LADDER.read_text())["rungs"]
# Miniature pipeline directory for each manifest entry. MiniMax H3 has no
# public miniature pipeline, so that entry stays an explicit skip.
MINIATURES: dict[str, str | None] = {
    "pixart-sigma-256": "pixart",
    "flux-1-dev": "flux",
    "qwen-image": "qwenimage",
    "wan-2.1-t2v-1.3b": "wan",
    "wan-2.1-i2v-14b-480p": "wan21-i2v",
    "wan-2.2-t2v-a14b": "wan22",
    "wan-2.2-i2v-a14b": "wan22-i2v",
    "minimax-h3": None,
}
# The miniature models are a few latent pixels wide; the manifest sizes are for
# the official checkpoints.
MINIATURE_SIZES = {"pixart": "256x256"}
MINIATURE_SIZE = "32x32"


def test_every_media_rung_is_accounted_for() -> None:
    assert {entry["model"] for entry in RUNGS} == set(MINIATURES)
    assert {entry["rung"] for entry in RUNGS} == {"M1", "M2", "M3", "M4"}


@pytest.mark.parametrize("entry", RUNGS, ids=[f"{e['rung']}-{e['model']}" for e in RUNGS])
def test_media_rung_executes_in_theory_with_its_pipeline_family(
    entry: dict[str, Any],
    tmp_path: Path,
    mtls_identities: dict[str, bytes],
) -> None:
    miniature = MINIATURES[entry["model"]]
    if miniature is None:
        pytest.skip(f"no public miniature pipeline exists for {entry['model']}")
    checkpoint = Path(FIXTURES) / miniature
    assert (checkpoint / "model_index.json").is_file()
    document = {key: value for key, value in entry.items() if key != "resolution_required"}
    document["workers"] = manifest_mapping()["workers"]
    document["checkpoint_fingerprint"] = "b" * 64
    document["verification"] = {
        **document.get("verification", {}),
        "size": MINIATURE_SIZES.get(miniature, MINIATURE_SIZE),
    }
    rung = LadderRung.from_mapping(document)
    definition = ModelDefinition(
        path=str(checkpoint), modality=rung.modality, engine="fortress", fortress_seed=731
    )
    adapter = MediaArchitectureAdapter.from_checkpoint(definition.path, rung.modality)
    compiled = compile_media_checkpoint(definition, adapter)
    workers: list[FortressRoleWorker] = []
    try:
        for index, placement in enumerate(rung.workers):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            worker = FortressRoleWorker(
                {
                    f"{component}:{name}": encode_role_artifact(
                        operation, placement.weight_index, public
                    )
                    for component, operations in compiled.items()
                    for name, operation in operations.items()
                },
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                allow_unattested=True,
            )
            worker.start()
            workers.append(worker)
            document["workers"][index]["endpoint"] = worker.address
            document["workers"][index]["recipient_public"] = public.hex()
        del compiled
        placed = LadderRung.from_mapping(document)
        for filename, key in (
            ("client.crt", "client_certificate"),
            ("client.key", "client_key"),
            ("ca.crt", "ca"),
        ):
            (tmp_path / filename).write_bytes(mtls_identities[key])
        sdk = sdk_configuration(
            placed,
            checkpoint_path=checkpoint,
            fortress_seed=731,
            certificate_path=tmp_path / "client.crt",
            key_path=tmp_path / "client.key",
            ca_path=tmp_path / "ca.crt",
        )
        report = verify_ladder({"rung": document, "sdk": sdk})
    finally:
        for worker in workers:
            worker.close()
    assert report["hardware_validated"] is False
    assert report["production_accepted"] is False
    assert report["quantized_reference_max_abs_error"] == 0
    assert report["original_model_max_abs_error"] >= 0
    assert report["bytes_sent"] > 0
    assert report["bytes_received"] > 0
