"""Tests for Martingale Theory module."""

import pytest
from metavoi.martingale import compute_martingale_voi


def test_e_process_nonnegative(bcg_input):
    """E-process values must be non-negative (product of positive e-values)."""
    result = compute_martingale_voi(bcg_input, n_sequential=10, n_per_trial=50, n_sims=100)
    for point in result["e_process_curve"]:
        assert point["mean_e"] >= 0.0, f"Negative mean E at t={point['t']}"
        assert point["median_e"] >= 0.0, f"Negative median E at t={point['t']}"


def test_stopping_time_positive(bcg_input):
    """Expected stopping time must be positive."""
    result = compute_martingale_voi(bcg_input, n_sequential=20, n_per_trial=50, n_sims=100)
    assert result["expected_stopping_time"] > 0.0


def test_grow_lambda_nonzero(bcg_input):
    """GROW optimal lambda should be non-zero when theta != mcid."""
    result = compute_martingale_voi(bcg_input, n_sequential=5, n_per_trial=50, n_sims=50)
    assert result["grow_lambda"] != 0.0


def test_anytime_threshold_correct(bcg_input):
    """Anytime-valid threshold should equal 1/alpha = 20."""
    result = compute_martingale_voi(bcg_input, n_sequential=5, n_per_trial=50, n_sims=50)
    assert abs(result["anytime_threshold"] - 20.0) < 1e-10


def test_safe_evsi_nonnegative(bcg_input):
    """Safe EVSI (EVSI adjusted for optional stopping) must be non-negative."""
    result = compute_martingale_voi(bcg_input, n_sequential=10, n_per_trial=50, n_sims=100)
    assert result["safe_evsi"] >= 0.0
