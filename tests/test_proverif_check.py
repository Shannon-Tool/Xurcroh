from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_proverif.py"


def run_checker(output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT)],
        input=output,
        capture_output=True,
        text=True,
        check=False,
    )


def test_proverif_checker_accepts_complete_true_summary() -> None:
    output = """Verification summary:

Query not attacker(activation_secret[]) is true.
Query not attacker(response_secret[]) is true.
Query event(ClientAccepted(m)) ==> event(ServerEstablished(m)) is true.
"""
    result = run_checker(output)
    assert result.returncode == 0, result.stderr
    assert result.stdout == output


def test_proverif_checker_rejects_false_query() -> None:
    result = run_checker(
        """Verification summary:

Query not attacker(activation_secret[]) is true.
Query not attacker(response_secret[]) is false.
"""
    )
    assert result.returncode == 1
    assert "response_secret" in result.stderr


def test_proverif_checker_rejects_missing_summary() -> None:
    result = run_checker("ProVerif terminated without a verification summary\n")
    assert result.returncode == 1
    assert "summary" in result.stderr
