#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from Xurcroh import Xurcroh
from xurcroh.errors import RequestCancelledError


def wait_for_workers(deployment: dict, timeout: float) -> list[dict]:
    deadline = time.monotonic() + timeout
    failures = {}
    while time.monotonic() < deadline:
        capabilities = []
        for worker in deployment["workers"]:
            try:
                with httpx.Client(verify=deployment["ca_file"], timeout=5) as client:
                    health = client.get(worker["endpoint"] + "/health")
                    health.raise_for_status()
                    response = client.get(worker["endpoint"] + "/v1/capabilities")
                    response.raise_for_status()
                    values = response.json()
                    if values.get("runtime") != "vllm-stage":
                        raise RuntimeError("worker does not report the vLLM stage runtime")
                    capabilities.append({"name": worker["name"], **values})
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                failures[worker["name"]] = type(error).__name__
                break
        if len(capabilities) == len(deployment["workers"]):
            return capabilities
        time.sleep(2)
    raise TimeoutError(f"workers did not become ready: {failures}")


def complete(client: Xurcroh, model: str, request_id: str, max_tokens: int) -> dict:
    started = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with a short technical sentence."}],
        temperature=0,
        max_tokens=max_tokens,
        request_id=request_id,
        timeout=300,
    )
    if response.execution.engine != "vllm-stages":
        raise RuntimeError(f"unexpected execution engine: {response.execution.engine}")
    return {
        "request_id": request_id,
        "wall_duration_ms": (time.monotonic() - started) * 1000,
        "response": response.model_dump(mode="json"),
    }


def run_validation(deployment: dict, max_tokens: int) -> dict:
    os.environ["SSL_CERT_FILE"] = str(Path(deployment["ca_file"]).resolve())
    with open(deployment["boundary_key_file"]) as handle:
        os.environ["XURCROH_CLIENT_BOUNDARY_KEY"] = handle.read().strip()
    capabilities = wait_for_workers(deployment, 30)
    workers = [
        {"name": worker["name"], "endpoint": worker["endpoint"]}
        for worker in deployment["workers"]
    ]
    with Xurcroh(
        cluster={"workers": workers, "pipeline_parallel": len(workers)},
        model_path={
            deployment["model"]: {
                "path": deployment["checkpoint"],
                "manifest_path": deployment["manifest"],
                "engine": "vllm",
            }
        },
        security={"mode": "strict"},
    ) as client:
        single = complete(client, deployment["model"], "validation-single", max_tokens)
        stream = client.chat.completions.create(
            model=deployment["model"],
            messages=[{"role": "user", "content": "Stream a short answer."}],
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            request_id="validation-stream",
            timeout=300,
        )
        chunks = [chunk.model_dump(mode="json") for chunk in stream]
        if not chunks or chunks[-1].get("usage") is None:
            raise RuntimeError("streaming response omitted terminal usage metadata")
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent = list(
                executor.map(
                    lambda index: complete(
                        client, deployment["model"], f"validation-concurrent-{index}", max_tokens
                    ),
                    range(2),
                )
            )
        cancelled = client.chat.completions.create(
            model=deployment["model"],
            messages=[{"role": "user", "content": "Generate a longer response."}],
            temperature=0,
            max_tokens=32,
            stream=True,
            request_id="validation-cancel",
            timeout=300,
        )
        cancellation = {"accepted": False, "completed": False}
        try:
            next(cancelled)
            cancellation["accepted"] = client.cancel("validation-cancel")
            for chunk in cancelled:
                if chunk.choices[0].finish_reason == "cancelled":
                    cancellation["completed"] = True
        except RequestCancelledError:
            cancellation["completed"] = True
        finally:
            cancelled.close()
        if not cancellation["accepted"] or not cancellation["completed"]:
            raise RuntimeError(f"cancellation did not complete: {cancellation}")
        recovered = complete(client, deployment["model"], "validation-after-cancel", 1)
        metrics = client.metrics()
    return {
        "schema_version": 1,
        "status": "passed",
        "revision": deployment["revision"],
        "execution": "public-sdk-vllm-stages",
        "worker_count": len(workers),
        "capabilities": capabilities,
        "single": single,
        "streaming_chunks": chunks,
        "concurrent": concurrent,
        "cancellation": cancellation,
        "recovered": recovered,
        "metrics": metrics,
        "hardware_validated": True,
        "production_accepted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment", type=Path)
    parser.add_argument("--wait-only", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=900)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    deployment = json.loads(args.deployment.read_text())
    if args.wait_only:
        print(json.dumps(wait_for_workers(deployment, args.wait_timeout), indent=2))
        return
    output = args.output or args.deployment.with_name("sdk-report.json")
    try:
        report = run_validation(deployment, args.max_tokens)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "revision": deployment.get("revision"),
            "error_type": type(error).__name__,
            "hardware_validated": False,
            "production_accepted": False,
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
