from __future__ import annotations

from pathlib import Path

import yaml


def test_fortress_worker_image_and_sandbox_are_sealed() -> None:
    root = Path(__file__).parents[2]
    dockerfile = (root / "deploy/fortress-worker.Dockerfile").read_text()
    assert "gcr.io/distroless/base-debian12:nonroot" in dockerfile
    assert "COPY --from=build /usr/local /usr/local" in dockerfile
    assert "COPY Cargo.toml Cargo.lock rust-toolchain.toml ./" in dockerfile
    assert "requirements-fortress.lock" in dockerfile
    assert "openssh" not in dockerfile
    assert "ENTRYPOINT" in dockerfile
    assert "xurcroh.fortress.worker_cli" in dockerfile
    assert "9444/udp" in dockerfile
    manifest = yaml.safe_load((root / "deploy/fortress-worker.yaml").read_text())
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert manifest["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    assert "@sha256:REPLACE_WITH_MEASURED_DIGEST" in container["image"]
    security = container["securityContext"]
    assert security["readOnlyRootFilesystem"] is True
    assert security["allowPrivilegeEscalation"] is False
    assert security["runAsNonRoot"] is True
    assert security["capabilities"]["drop"] == ["ALL"]
    assert security["seccompProfile"]["type"] == "RuntimeDefault"
    assert container["ports"] == [{"name": "quic", "containerPort": 9444, "protocol": "UDP"}]
    assert "xurcroh.fortress.worker_cli" in container["command"]
    assert container["readinessProbe"]["exec"]["command"][0] == "/opt/venv/bin/python"
