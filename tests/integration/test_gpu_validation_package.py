from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import yaml
from cryptography import x509

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_package_is_pinned_and_records_hardware() -> None:
    lock = (ROOT / "requirements-cuda.lock").read_text().splitlines()
    assert "torch==2.9.1" in lock
    assert "transformers==4.57.6" in lock
    assert "vllm==0.15.0" in lock
    assert "pytest==9.2.0" in lock
    assert "pytest-asyncio==1.4.1.post1" in lock
    bootstrap = (ROOT / "scripts/gpu/bootstrap.sh").read_text()
    assert '"$PYTHON" -m venv --system-site-packages "$VENV"' in bootstrap
    assert 'VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop' in bootstrap
    assert "export LD_LIBRARY_PATH=" in bootstrap
    assert "/usr/local/nvidia/lib64" in bootstrap
    assert "--retry-all-errors" in bootstrap
    assert "vendor_id" in bootstrap
    assert "--retries 10 --timeout 120" in bootstrap
    compose = yaml.safe_load((ROOT / "deploy/cuda/compose.yaml").read_text())
    for index, name in enumerate(("worker-a", "worker-b")):
        worker = compose["services"][name]
        assert f"--stage-index={index}" in worker["command"]
        assert any(f"/stage-{index}:/artifacts:ro" in volume for volume in worker["volumes"])
        assert any(f"identities/stage-{index}.key" in volume for volume in worker["volumes"])
        assert worker["environment"]["RANK"] == str(index)
        assert worker["environment"]["WORLD_SIZE"] == "2"
        assert worker["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 1


def test_checkpoint_compilation_prepares_worker_only_artifacts(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "compiled"
    subprocess.run(  # noqa: S603
        [
            str(ROOT / "scripts/gpu/compile_checkpoint.sh"),
            str(tiny_llama_path),
            str(output),
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not list((output / "worker-checkpoint").glob("*.safetensors"))
    assert (output / "worker-checkpoint/config.json").is_file()
    for index in range(2):
        files = {path.name for path in (output / f"stage-{index}").iterdir()}
        assert files == {"manifest.json", f"stage-{index}.xurcroh", f"stage-{index}.key.json"}
        mode = stat.S_IMODE((output / f"identities/stage-{index}.key").stat().st_mode)
        assert mode == 0o600
    assert stat.S_IMODE((output / "client/boundary-key.hex").stat().st_mode) == 0o600


def test_tls_generator_creates_private_identities_and_san_certificates(tmp_path: Path) -> None:
    module_path = ROOT / "scripts/gpu/generate_tls.py"
    spec = importlib.util.spec_from_file_location("generate_tls", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "tls"
    module.generate(output, {"worker-a": ["192.0.2.10", "worker-a"]})
    certificate = x509.load_pem_x509_certificate((output / "worker-a.crt").read_bytes())
    names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert names.get_values_for_type(x509.IPAddress)
    assert "worker-a" in names.get_values_for_type(x509.DNSName)
    assert stat.S_IMODE((output / "worker-a.key").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "ca.key").stat().st_mode) == 0o600


def test_report_preserves_failed_validation_and_destruction_state(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk.json"
    sdk.write_text(json.dumps({"status": "failed"}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"total_cost_usd": 1.25}))
    destruction = tmp_path / "destruction.json"
    destruction.write_text(json.dumps({"all_destroyed": False}))
    output = tmp_path / "report.json"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/gpu/report.py"),
            "--sdk-report",
            str(sdk),
            "--ledger",
            str(ledger),
            "--destruction",
            str(destruction),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    report = json.loads(output.read_text())
    assert report["hardware_validated"] is False
    assert report["production_accepted"] is False
    assert report["spending"]["total_cost_usd"] == 1.25
    assert report["destruction"]["all_destroyed"] is False
