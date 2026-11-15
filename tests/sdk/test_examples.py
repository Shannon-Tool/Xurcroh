from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_managed_gemma_example_exposes_cli_without_starting_inference() -> None:
    example = Path(__file__).parents[2] / "examples" / "managed_gemma.py"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(example), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "managed Fortress grid" in result.stdout
    assert "--datagram-tunnel" in result.stdout


def test_managed_gemma_example_rejects_incomplete_grid() -> None:
    example = Path(__file__).parents[2] / "examples" / "managed_gemma.py"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(example), "--host", "localhost"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "exactly 4" in result.stderr


def test_managed_gemma_example_computes_larger_grid_size() -> None:
    example = Path(__file__).parents[2] / "examples" / "managed_gemma.py"
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(example), "--shares", "3", "--host", "localhost"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "exactly 9" in result.stderr
