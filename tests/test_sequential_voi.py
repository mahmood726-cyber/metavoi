"""Tests for sequential VoI (adaptive evidence acquisition)."""

import pytest
from metavoi.sequential_voi import compute_sequential_voi


def test_sequential_value_gte_single_stage(bcg_input):
    """Sequential value of waiting must be >= 0 (at least as good as decide-now)."""
    result = compute_sequential_voi(bcg_input, n_per_stage=500, T=3, n_mc=200)
    assert result["value_of_waiting"] >= 0.0


def test_sequential_strategy_length(bcg_input):
    """Strategy list has T entries."""
    T = 3
    result = compute_sequential_voi(bcg_input, n_per_stage=500, T=T, n_mc=200)
    assert len(result["optimal_strategy"]) == T
    assert len(result["stage_values"]) == T


def test_sequential_strategy_valid_actions(bcg_input):
    """Each action is one of: 'decide', 'trial', 'n/a'."""
    result = compute_sequential_voi(bcg_input, n_per_stage=500, T=3, n_mc=200)
    for action in result["optimal_strategy"]:
        assert action in ("decide", "trial", "n/a")


def test_sequential_certain_input_decides_immediately(certain_input):
    """Very certain evidence -> decide at stage 0, no trial needed."""
    result = compute_sequential_voi(certain_input, n_per_stage=1000, T=3, n_mc=200)
    assert result["optimal_strategy"][0] == "decide"
    assert result["value_of_waiting"] < 0.01


def test_sequential_deterministic(bcg_input):
    """Same seed -> same result."""
    r1 = compute_sequential_voi(bcg_input, n_per_stage=500, T=2, n_mc=100)
    r2 = compute_sequential_voi(bcg_input, n_per_stage=500, T=2, n_mc=100)
    assert r1["expected_value"] == r2["expected_value"]
    assert r1["optimal_strategy"] == r2["optimal_strategy"]
