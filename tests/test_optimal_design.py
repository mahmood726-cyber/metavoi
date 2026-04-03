"""Tests for Optimal Design module."""

import pytest
from metavoi.optimal_design import compute_optimal_design


def test_d_optimal_n_positive(bcg_input):
    """D-optimal sample size must be a positive integer."""
    result = compute_optimal_design(bcg_input)
    assert isinstance(result["d_optimal_n"], int)
    assert result["d_optimal_n"] > 0


def test_a_optimal_n_positive(bcg_input):
    """A-optimal sample size must be a positive integer."""
    result = compute_optimal_design(bcg_input)
    assert isinstance(result["a_optimal_n"], int)
    assert result["a_optimal_n"] > 0


def test_multi_site_respects_budget(bcg_input):
    """Total multi-site cost must not exceed budget."""
    budget = bcg_input.cost_per_patient * 5000
    result = compute_optimal_design(bcg_input, budget=budget)
    total_cost = sum(s["cost"] for s in result["multi_site_allocation"])
    assert total_cost <= budget + 1e-6


def test_info_gain_curve_monotonic(bcg_input):
    """Information gain should be non-decreasing with N."""
    result = compute_optimal_design(bcg_input)
    curve = result["info_gain_curve"]
    for i in range(1, len(curve)):
        assert curve[i]["delta_info"] >= curve[i - 1]["delta_info"] - 1e-12


def test_comparison_has_all_methods(bcg_input):
    """Comparison dict should contain all three optimization criteria."""
    result = compute_optimal_design(bcg_input)
    assert "d_optimal" in result["comparison"]
    assert "a_optimal" in result["comparison"]
    assert "evsi_optimal" in result["comparison"]
    for method, n in result["comparison"].items():
        assert isinstance(n, int) and n > 0
