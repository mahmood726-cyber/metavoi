"""Tests for Bayesian Bootstrap module."""

import pytest
from metavoi.bayesian_bootstrap import compute_bayesian_bootstrap
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_bootstrap_ci_contains_point_estimate(bcg_input):
    """Bootstrap 95% CI should contain the standard EVPI point estimate."""
    draws = predictive_distribution(bcg_input)
    evpi_point = compute_evpi(draws, bcg_input.mcid)
    result = compute_bayesian_bootstrap(bcg_input, n_boot=200, n_mc=5000)
    lo, hi = result["evpi_ci"]
    assert lo <= evpi_point <= hi


def test_evpi_cv_positive(bcg_input):
    """Coefficient of variation must be positive for uncertain data."""
    result = compute_bayesian_bootstrap(bcg_input, n_boot=200, n_mc=5000)
    assert result["evpi_cv"] > 0


def test_distribution_length_matches_n_boot(bcg_input):
    """Bootstrap distribution should have exactly n_boot samples."""
    n_boot = 100
    result = compute_bayesian_bootstrap(bcg_input, n_boot=n_boot, n_mc=3000)
    assert len(result["evpi_distribution"]) == n_boot


def test_p_justified_between_zero_and_one(bcg_input):
    """Probability of justification must be in [0, 1]."""
    result = compute_bayesian_bootstrap(bcg_input, n_boot=100, n_mc=3000)
    assert 0.0 <= result["p_justified"] <= 1.0


def test_evsi_ci_nonnegative(bcg_input):
    """EVSI credible interval bounds should be non-negative."""
    result = compute_bayesian_bootstrap(bcg_input, n_boot=100, n_mc=3000,
                                        n_trial_evsi=500)
    lo, hi = result["evsi_ci"]
    assert lo >= -1e-6
    assert hi >= lo
