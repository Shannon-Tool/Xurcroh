from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tests.fortress.test_ladder import evidence_mapping, manifest_mapping

from xurcroh.fortress.ladder import (
    LadderEvidenceError,
    LadderRung,
    verify_rung_evidence,
    write_rung_evidence,
)


def valid_v2_report() -> dict:
    return {
        **evidence_mapping(),
        "evidence_version": 2,
        "source_format": "exl3",
        "source_digest": "c" * 64,
        "architecture_adapter": "llama",
        "decoder_identity": {"package": "exllamav3", "version": "1.6.7", "revision": "d" * 64},
        "source_decoder_max_abs_error": 0.0,
        "official_decoder_validated": True,
        "workflow": "text",
        "client_resources": {"peak_rss_bytes": 1024},
        "role_resources": {"role-0-0": {"peak_vram_bytes": 2048}},
        "phase_seconds": {"verification": 1.0},
        "relay_mode": "direct",
        "result": "pass",
    }


def test_v2_requires_format_and_official_decoder_provenance(tmp_path: Path) -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    private = Ed25519PrivateKey.generate()
    report = valid_v2_report()
    del report["decoder_identity"]
    with pytest.raises(LadderEvidenceError, match="decoder_identity"):
        write_rung_evidence(tmp_path / "invalid.json", rung, report, signing_key=private)
    write_rung_evidence(
        tmp_path / "valid.json", rung, valid_v2_report(), signing_key=private, format_version=2
    )
    _, verified = verify_rung_evidence(tmp_path / "valid.json", public_key=private.public_key())
    assert verified["source_format"] == "exl3"
    assert verified["official_decoder_validated"] is True


def test_v2_hardware_rung_rejects_unverified_official_decoder(tmp_path: Path) -> None:
    rung = LadderRung.from_mapping(manifest_mapping())
    report = valid_v2_report()
    report["official_decoder_validated"] = False
    with pytest.raises(LadderEvidenceError, match="official decoder"):
        write_rung_evidence(
            tmp_path / "invalid.json", rung, report, signing_key=Ed25519PrivateKey.generate()
        )
