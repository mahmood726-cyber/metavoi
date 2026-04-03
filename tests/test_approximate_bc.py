"""Tests for Approximate Bayesian Computation VoI module."""

import numpy as np
import pytest
from metavoi.approximate_bc import compute_abc_voi


def test_abc_mean_within_3se_of_analytic(bcg_input):
    """ABC posterior mean should be within 3 * analytic SE of the true theta."""
    result = compute_abc_voi(bcg_input)
    pred_se = np.sqrt(bcg_input.se ** 2 + bcg_input.tau2)
    assert abs(result["abc_posterior_mean"] - bcg_input.theta) < 3 * pred_se, (
        f"ABC mean {result['abc_posterior_mean']} too far from theta {bcg_input.theta}"
    )


def test_abc_posterior_sd_positive(bcg_input):
    """ABC posterior SD must be positive."""
    result = compute_abc_voi(bcg_input)
    assert result["abc_posterior_sd"] > 0.0


def test_abc_evpi_tolerance_schedule(bcg_input):
    """ABC EVPI must be reported for all 4 tolerance levels."""
    result = compute_abc_voi(bcg_input)
    tols = [pt["tolerance"] for pt in result["abc_evpi_by_tolerance"]]
    assert tols == [2.0, 1.0, 0.5, 0.2]


def test_acceptance_rate_decreasing(bcg_input):
    """Acceptance rate should generally decrease with tighter tolerance."""
    result = compute_abc_voi(bcg_input)
    rates = [pt["acceptance_rate"] for pt in result["abc_evpi_by_tolerance"]]
    # Each tighter tolerance should accept fewer or equal proposals
    for i in range(1, len(rates)):
        assert rates[i] <= rates[i - 1] + 1e-10, (
            f"Acceptance rate increased from tol={result['abc_evpi_by_tolerance'][i-1]['tolerance']} "
            f"to tol={result['abc_evpi_by_tolerance'][i]['tolerance']}"
        )


def test_bayes_factor_positive(bcg_input):
    """Bayes factor RE vs FE must be positive."""
    result = compute_abc_voi(bcg_input)
    assert result["bayes_factor_re_vs_fe"] > 0.0
