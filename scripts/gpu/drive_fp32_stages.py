#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: S607, S108
import argparse
import json
import subprocess
from pathlib import Path

import deploy_two_node as deploy
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from xurcroh.client.boundary import ClientBoundary

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-prefix", default="/tmp/xurcroh-fp32-stage")
    parser.add_argument("--boundary-key", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if len(config["workers"]) != 1:
        raise ValueError("FP32 sequential validation requires exactly one worker")
    worker = config["workers"][0]
    manifest = json.loads((args.compiled / "manifest.json").read_text())
    stage_count = int(manifest["stage_count"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    key_path = args.boundary_key or args.compiled / "client" / "boundary-key.hex"
    boundary = ClientBoundary(
        args.compiled / "client-boundary.xurcroh", bytes.fromhex(key_path.read_text().strip())
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    tokens = tokenizer("user: Reply with a short technical sentence.", return_tensors="pt")[
        "input_ids"
    ]
    hidden = boundary.embed(tokens)[0]
    decode_hidden = boundary.embed(tokens[:, -1:])[0]
    residual = None
    decode_residual = None
    reports = []
    try:
        for index in range(stage_count):
            if index > 0:
                hidden = boundary.transition(hidden, index - 1)
                decode_hidden = boundary.transition(decode_hidden, index - 1)
                if residual is not None:
                    residual = boundary.transition(residual, index - 1)
                    decode_residual = boundary.transition(decode_residual, index - 1)
            inputs = {"input-0": hidden, "input-1": decode_hidden}
            if residual is not None:
                inputs["residual-0"] = residual
                inputs["residual-1"] = decode_residual
            input_path = args.output / f"pipeline-input-{index}.safetensors"
            save_file(inputs, str(input_path))
            remote_root = f"{args.remote_prefix}-{index}"
            deploy.scp(worker, [input_path], f"/tmp/xurcroh-pipeline-input-{index}.safetensors")
            deploy.scp(
                worker,
                [Path(__file__).with_name("validate_fp32_stage.py")],
                "/tmp/xurcroh-validate-fp32-stage.py",
            )
            command = (
                f"cd {remote_root} && "
                "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:"
                "/usr/local/nvidia/lib64:/usr/local/cuda/lib64 && "
                f"export PYTHONPATH={remote_root}/python && "
                f".venv-cuda/bin/python /tmp/xurcroh-validate-fp32-stage.py "
                f"--root {remote_root} --index {index} "
                f"--input /tmp/xurcroh-pipeline-input-{index}.safetensors "
                f"--output /tmp/xurcroh-pipeline-output-{index}.safetensors "
                f"--report /tmp/xurcroh-stage-{index}-numerical.json"
            )
            deploy.ssh(worker, command)
            report_path = args.output / f"stage-{index}-numerical.json"
            deploy.download(worker, f"/tmp/xurcroh-stage-{index}-numerical.json", report_path)
            reports.append(json.loads(report_path.read_text()))
            output_path = args.output / f"pipeline-output-{index}.safetensors"
            deploy.download(
                worker, f"/tmp/xurcroh-pipeline-output-{index}.safetensors", output_path
            )
            output = load_file(str(output_path))
            hidden = output["forward-0"]
            decode_hidden = output["forward-1"]
            residual = output.get("residual-0")
            decode_residual = output.get("residual-1")
            print(json.dumps(reports[-1]), flush=True)
    finally:
        boundary.close()

    passed = len(reports) == stage_count and all(item["status"] == "passed" for item in reports)
    summary = {
        "revision": revision,
        "status": "passed" if passed else "failed",
        "stage_count": len(reports),
        "worker_count": 1,
        "execution": "sequential-vllm-stages-on-one-gpu",
        "reports": reports,
        "hardware_validated": passed,
        "production_accepted": False,
        "distributed_validated": False,
    }
    report_path = args.output / "fp32-stage-validation.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "stage_count": len(reports)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
