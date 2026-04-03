"""Tests for stochastic dominance analysis module."""

import pytest
from metavoi.stochastic_dominance import compute_stochastic_dominance


def test_fsd_ratio_in_range(bcg_input):
    """FSD ratio must be in [0, 1]."""
    result = compute_stochastic_dominance(bcg_input)
    assert 0.0 <= result["fsd_ratio"] <= 1.0


def test_ssd_implied_by_fsd(bcg_input):
    """If FSD holds (dominates), SSD must also hold (weaker condition)."""
    result = compute_stochastic_dominance(bcg_input)
    if result["fsd_treat_dominates"]:
        assert result["ssd_treat_dominates"]
    # Both ratios must be in [0, 1]
    assert 0.0 <= result["ssd_ratio"] <= 1.0


def test_gini_in_range(bcg_input):
    """Gini coefficients must be in [0, 1]."""
    result = compute_stochastic_dominance(bcg_input)
    assert 0.0 <= result["gini_treat"] <= 1.0
    assert 0.0 <= result["gini_no_treat"] <= 1.0


def test_cvar_leq_var(bcg_input):
    """CVaR (expected shortfall) should be <= VaR (by definition, for losses)."""
    result = compute_stochastic_dominance(bcg_input)
    # CVaR is the mean of the worst alpha fraction, VaR is the alpha quantile.
    # For the worst-case tail, CVaR <= VaR.
    assert result["cvar_treat"] <= result["var_treat"] + 1e-9
    assert result["cvar_no_treat"] <= result["var_no_treat"] + 1e-9


def test_recommendation_is_string(bcg_input):
    """Recommendation must be a non-empty string."""
    result = compute_stochastic_dominance(bcg_input)
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0
