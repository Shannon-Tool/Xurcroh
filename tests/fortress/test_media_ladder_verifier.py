from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.test_ladder import manifest_mapping

from xurcroh.fortress.diffusers_loader import compile_diffusers_component
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import sdk_configuration
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker import FortressRoleWorker


def test_image_ladder_verifier_compares_latents_and_public_sdk(
    tmp_path: Path, mtls_identities: dict[str, bytes]
) -> None:
    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    pipeline = diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler())
    pipeline.save_pretrained(tmp_path / "model")
    mutable = manifest_mapping()
    mutable["model"] = "tiny-image"
    mutable["modality"] = "image"
    rung = LadderRung.from_mapping(mutable)
    operations = compile_diffusers_component(tmp_path / "model", component="unet", seed=731)
    workers: list[FortressRoleWorker] = []
    try:
        for index, placement in enumerate(rung.workers):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            artifacts = {
                name: encode_role_artifact(operation, placement.weight_index, public)
                for name, operation in operations.items()
            }
            worker = FortressRoleWorker(
                artifacts,
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
            )
            worker.start()
            workers.append(worker)
            mutable["workers"][index]["endpoint"] = worker.address
            mutable["workers"][index]["recipient_public"] = public.hex()
        actual_rung = LadderRung.from_mapping(mutable)
        for filename, key in (
            ("client.crt", "client_certificate"),
            ("client.key", "client_key"),
            ("ca.crt", "ca"),
        ):
            (tmp_path / filename).write_bytes(mtls_identities[key])
        sdk = sdk_configuration(
            actual_rung,
            checkpoint_path=tmp_path / "model",
            fortress_seed=731,
            certificate_path=tmp_path / "client.crt",
            key_path=tmp_path / "client.key",
            ca_path=tmp_path / "ca.crt",
        )
        report = verify_ladder({"rung": mutable, "sdk": sdk})
        assert report["quantized_reference_max_abs_error"] == 0
        assert report["original_model_max_abs_error"] >= 0
        assert report["sdk"]["response"]["data"][0]["b64_json"]
    finally:
        for worker in workers:
            worker.close()
