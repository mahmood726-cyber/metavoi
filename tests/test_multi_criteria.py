"""Tests for multi-criteria decision analysis VoI module."""

import pytest
from metavoi.multi_criteria import compute_multi_criteria


def test_weighted_evpi_nonnegative(bcg_input):
    """Weighted EVPI must be >= 0."""
    result = compute_multi_criteria(bcg_input)
    assert result["weighted_evpi"] >= 0.0


def test_topsis_scores_in_range(bcg_input):
    """All TOPSIS scores must be in [0, 1]."""
    result = compute_multi_criteria(bcg_input)
    for entry in result["topsis_ranking"]:
        assert 0.0 <= entry["score"] <= 1.0, (
            f"TOPSIS score {entry['score']} out of [0,1] for n={entry['n']}"
        )


def test_pareto_optimal_subset_of_n_values(bcg_input):
    """Pareto-optimal N values must be a subset of evaluated N values."""
    n_values = [50, 100, 200, 500, 1000, 2000, 5000]
    result = compute_multi_criteria(bcg_input, n_values=n_values)
    for n in result["pareto_optimal_n"]:
        assert n in n_values


def test_default_outcomes_created(bcg_input):
    """When no outcomes specified, should create efficacy + safety."""
    result = compute_multi_criteria(bcg_input)
    assert "efficacy" in result["outcome_names"]
    assert "safety" in result["outcome_names"]
    assert "efficacy" in result["per_outcome_evpi"]
    assert "safety" in result["per_outcome_evpi"]


def test_best_n_topsis_in_ranking(bcg_input):
    """best_n_topsis should be the top-ranked N from TOPSIS."""
    result = compute_multi_criteria(bcg_input)
    best = result["best_n_topsis"]
    ranking = result["topsis_ranking"]
    assert ranking[0]["n"] == best
