from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from xurcroh.fortress.arithmetic import FIELD_MODULUS
from xurcroh.fortress.native import FortressNativeController

pytestmark = pytest.mark.hardware


def require_cuda() -> None:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        pytest.skip("CUDA compiler and GPU are required for Fortress CUDA validation")
    executable = shutil.which("nvidia-smi")
    assert executable is not None
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

    assert controller.matvec([[1, 2], [3, 4]], [5, 7]) == [26, 38]
    first = controller.chacha_words(list(range(8)), bytes(12), 32)
    second = controller.chacha_words(list(range(8)), bytes(12), 32)
    assert first == second
    assert any(first)


def _chacha_reference(key: list[int], nonce: bytes, count: int) -> list[int]:
    def rotate(value: int, shift: int) -> int:
        return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF

    def quarter(state: list[int], a: int, b: int, c: int, d: int) -> None:
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = rotate(state[d] ^ state[a], 16)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = rotate(state[b] ^ state[c], 12)
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = rotate(state[d] ^ state[a], 8)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = rotate(state[b] ^ state[c], 7)

    output: list[int] = []
    for block in range((count + 15) // 16):
        state = [
            0x61707865,
            0x3320646E,
            0x79622D32,
            0x6B206574,
            *key,
            block,
            *(int.from_bytes(nonce[index : index + 4], "little") for index in (0, 4, 8)),
        ]
        working = state.copy()
        for _ in range(10):
            for indexes in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)):
                quarter(working, *indexes)
            for indexes in ((0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
                quarter(working, *indexes)
        output.extend(
            (left + right) & 0xFFFFFFFF for left, right in zip(state, working, strict=True)
        )
    return output[:count]


@pytest.mark.parametrize(
    "weight,activation",
    [
        (804968700, 3266489909),
        (0, FIELD_MODULUS - 1),
        (1, FIELD_MODULUS - 1),
        (FIELD_MODULUS - 1, FIELD_MODULUS - 1),
        (FIELD_MODULUS - 2, FIELD_MODULUS - 3),
        (2**31 + 17, 2**31 + 19),
        (65535, 65537),
    ],
)
def test_field_kernel_matches_torch_single_product(weight: int, activation: int) -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    controller = FortressNativeController(root / "native/build/xurcroh_burn_alloc.so")
    product = weight * activation
    expected = product % FIELD_MODULUS
    if product <= torch.iinfo(torch.int64).max:
        torch_result = (
            torch.tensor(weight, dtype=torch.int64, device="cuda")
            * torch.tensor(activation, dtype=torch.int64, device="cuda")
        ).remainder(FIELD_MODULUS)
        assert int(torch_result.cpu().item()) == expected
    assert controller.matvec([[weight]], [activation]) == [expected]


@pytest.mark.parametrize("rows,cols", [(1, 1), (7, 5), (65, 33), (257, 129), (1024, 768)])
def test_integer_matvec_matches_field_reference(rows: int, cols: int) -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    controller = FortressNativeController(root / "native/build/xurcroh_burn_alloc.so")
    weights = [
        [((row + 1) * 0x9E3779B1 + (col + 3) * 0x85EBCA6B) % FIELD_MODULUS for col in range(cols)]
        for row in range(rows)
    ]
    activations = [((index + 1) * 0xC2B2AE35) % FIELD_MODULUS for index in range(rows)]
    expected = [
        sum(weights[row][col] * activations[row] for row in range(rows)) % FIELD_MODULUS
        for col in range(cols)
    ]
    assert controller.matvec(weights, activations) == expected


def test_chacha_expansion_matches_cpu_reference() -> None:
    require_cuda()
    root = Path(__file__).parents[2]
    controller = FortressNativeController(root / "native/build/xurcroh_burn_alloc.so")
    key = [0x03020100 + index * 0x04040404 for index in range(8)]
    nonce = bytes(range(12))
    assert controller.chacha_words(key, nonce, 80) == _chacha_reference(key, nonce, 80)
