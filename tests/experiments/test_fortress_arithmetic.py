from __future__ import annotations

from experiments.fortress_arithmetic import benchmark


def test_fortress_arithmetic_benchmark_is_exact_and_labels_security() -> None:
    result = benchmark(iterations=2)

    assert result["exact_reconstruction"] is True
    assert result["full_share_storage_bytes"] > result["seed_expanded_storage_bytes"]
    assert result["storage_savings_bytes"] > 0
    assert result["full_share_security"] == "information-theoretic"
    assert result["seed_expanded_security"] == "computational"
    assert result["full_split_ms"] >= 0
    assert result["seeded_split_ms"] >= 0
    assert result["prime_challenge_ms"] >= 0
