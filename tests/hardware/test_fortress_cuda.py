from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from xurcroh.fortress.native import FortressNativeController

pytestmark = pytest.mark.hardware


def require_cuda() -> None:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        pytest.skip("CUDA compiler and GPU are required for Fortress CUDA validation")
    executable = shutil.which("nvidia-smi")
    result = subprocess.run(  # noqa: S603
        [executable, "--query-gpu=count", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or int(result.stdout.strip() or "0") < 1:
        pytest.skip("CUDA-capable GPU is required for Fortress CUDA validation")


def test_integer_matvec_and_chacha_expansion() -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    subprocess.run([str(root / "scripts/build-native.sh")], check=True)  # noqa: S603
    controller = FortressNativeController(root / "native/build/xurcroh_burn_alloc.so")

    assert controller.matvec([[1, 2], [3, 4]], [5, 7]) == [19, 43]
    first = controller.chacha_words(list(range(8)), bytes(12), 32)
    second = controller.chacha_words(list(range(8)), bytes(12), 32)
    assert first == second
    assert any(first)
