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

from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import sdk_configuration
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker import FortressRoleWorker


def test_ladder_verifier_compares_remote_tensors_and_public_sdk(
    tiny_llama_path: Path,
    tmp_path: Path,
    mtls_identities: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc

    from transformers import AutoModelForCausalLM

    events: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))
    original_loader = AutoModelForCausalLM.from_pretrained

    def tracked_loader(*args, **kwargs):
        events.append("original")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", tracked_loader)
    rung = LadderRung.from_mapping(manifest_mapping())
    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    workers: list[FortressRoleWorker] = []
    mutable = manifest_mapping()
    try:
        for index, placement in enumerate(rung.workers):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            artifacts = {
                name: encode_role_artifact(operation, placement.weight_index, public)
                for name, operation in compiled.operations.items()
            }
            worker = FortressRoleWorker(
                artifacts,
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                allow_unattested=True,
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
            checkpoint_path=tiny_llama_path,
            fortress_seed=731,
            certificate_path=tmp_path / "client.crt",
            key_path=tmp_path / "client.key",
            ca_path=tmp_path / "ca.crt",
        )
        report = verify_ladder({"rung": mutable, "sdk": sdk})
        assert report["quantized_reference_max_abs_error"] == 0
        assert report["original_model_max_abs_error"] >= 0
        assert report["sdk"]["response"]["execution"]["engine"] == "fortress-grid"
        assert report["bytes_sent"] > 0
        assert report["bytes_received"] > 0
        assert report["attack_results"]["full_weight_collusion_exact"] is True
        assert report["canary_tripped"] is True
        assert report["freivalds_failures"] == 0
        assert report["lifecycle"]["streaming_completion_tokens"] == 2
        assert report["lifecycle"]["cancelled"] is True
        assert report["lifecycle"]["post_cancellation_completion_tokens"] == 1
        assert report["lifecycle"]["concurrent_requests"] == 2
        last_original = max(index for index, event in enumerate(events) if event == "original")
        assert "gc" in events[:last_original]
        assert report["client_phase_rss_bytes"]["original_model"] > 0
    finally:
        for worker in workers:
            worker.close()


def test_media_workflows_include_masked_editing_pipeline() -> None:
    from xurcroh.fortress.ladder_verifier import _media_workflows

    document = manifest_mapping("M2")
    document["modality"] = "image"
    document["verification"] = {"workflow_pipelines": {"masked_edit": "CustomInpaintPipeline"}}
    rung = LadderRung.from_mapping(document)

    class Pipeline:
        transformer = object()

        def __call__(self, prompt, image=None, mask_image=None):
            return None

    workflows = _media_workflows(rung, Pipeline())
    assert {workflow.name for workflow in workflows} == {
        "text_to_image",
        "image_to_image",
        "masked_edit",
    }
    selected = {workflow.name: workflow for workflow in workflows}
    assert selected["masked_edit"].pipeline_class == "CustomInpaintPipeline"
    assert selected["masked_edit"].masked is True
