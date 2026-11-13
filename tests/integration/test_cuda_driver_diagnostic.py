from __future__ import annotations

from scripts.fortress.cuda_driver_diagnostic import is_compat_library, kernel_driver_version


def test_driver_diagnostic_parses_kernel_version_and_compat_paths() -> None:
    version = kernel_driver_version("NVRM version: NVIDIA UNIX Kernel Module 580.159.03")
    assert version == "580.159.03"
    assert is_compat_library("/usr/local/cuda-12.9/compat/libcuda.so.1")
    assert not is_compat_library("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
