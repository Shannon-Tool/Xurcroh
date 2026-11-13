#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def kernel_driver_version(text: str) -> str | None:
    match = re.search(r"\d+(?:\.\d+)+", text)
    return match.group(0) if match else None


def is_compat_library(path: str) -> bool:
    return "/compat/" in path


def collect() -> dict[str, Any]:
    kernel_path = Path("/proc/driver/nvidia/version")
    kernel_text = kernel_path.read_text() if kernel_path.is_file() else ""
    kernel = kernel_driver_version(kernel_text)
    nvidia_smi = shutil.which("nvidia-smi")
    nvml = None
    if nvidia_smi is not None:
        result = subprocess.run(  # noqa: S603
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            nvml = result.stdout.strip().splitlines()[0]
    report: dict[str, Any] = {
        "kernel_driver": kernel,
        "nvml_driver": nvml,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
            )
        },
    }
    try:
        library = ctypes.CDLL("libcuda.so.1")
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.cuDeviceGetCount.restype = ctypes.c_int
        mapped = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "libcuda" in line
            }
        )
        init_status = library.cuInit(0)
        count = ctypes.c_int(-1)
        count_status = library.cuDeviceGetCount(ctypes.byref(count))
        report.update(
            {
                "mapped_libcuda": mapped,
                "compat_libcuda": any(is_compat_library(path) for path in mapped),
                "cu_init_status": init_status,
                "cu_device_count_status": count_status,
                "cu_device_count": count.value,
            }
        )
    except (OSError, AttributeError) as error:
        report["driver_error"] = f"{type(error).__name__}: {error}"
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_available"] = torch.cuda.is_available()
        if report["torch_cuda_available"]:
            torch.zeros(1, device="cuda")
            report["torch_tensor_allocation"] = "passed"
        else:
            report["torch_tensor_allocation"] = "not_attempted"
    except Exception as error:
        report["torch_error"] = f"{type(error).__name__}: {error}"
    report["driver_match"] = kernel is not None and kernel == nvml
    report["passed"] = (
        report["driver_match"]
        and report.get("compat_libcuda") is False
        and report.get("cu_init_status") == 0
        and report.get("cu_device_count_status") == 0
        and report.get("cu_device_count", 0) > 0
        and report.get("torch_tensor_allocation") == "passed"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CUDA driver and loader compatibility")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
