"""Tests for variational Bayes VoI module."""

import pytest
from metavoi.variational_bayes import compute_variational_bayes


def test_elbo_monotonically_increases(bcg_input):
    """CAVI should produce a non-decreasing ELBO trace."""
    result = compute_variational_bayes(bcg_input)
    trace = result["elbo_trace"]
    assert len(trace) >= 2
    for i in range(1, len(trace)):
        # Allow tiny floating-point jitter (1e-8)
        assert trace[i] >= trace[i - 1] - 1e-8, (
            f"ELBO decreased at step {i}: {trace[i]} < {trace[i-1]}"
        )


def test_convergence_flag(bcg_input):
    """BCG input should converge within 200 iterations."""
    result = compute_variational_bayes(bcg_input)
    assert result["converged"] is True
    assert result["iterations"] <= 200


def test_vb_evpi_nonnegative(bcg_input):
    """VB-EVPI must be non-negative and finite."""
    result = compute_variational_bayes(bcg_input)
    assert result["vb_evpi"] >= 0.0
    assert result["mc_evpi"] >= 0.0


def test_variational_params_reasonable(bcg_input):
    """Variational parameters should be in sensible ranges."""
    result = compute_variational_bayes(bcg_input)
    # mu_q should be near the observed theta
    assert abs(result["mu_q"] - bcg_input.theta) < 1.0
    # sigma_q should be positive and smaller than prior SD (sqrt(100)=10)
    assert 0 < result["sigma_q"] < 10.0
    # InvGamma shape a_q > 0, rate b_q > 0
    assert result["a_q"] > 0
    assert result["b_q"] > 0


def test_kl_divergence_nonneg(bcg_input):
    """KL divergence approximation should be non-negative."""
    result = compute_variational_bayes(bcg_input)
    assert result["kl_divergence_approx"] >= 0.0
