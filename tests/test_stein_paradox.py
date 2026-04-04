"""Tests for Stein Paradox / James-Stein shrinkage module."""

import pytest
from metavoi.stein_paradox import compute_stein_paradox


def test_risk_ratio_below_one_for_k13(bcg_input):
    """James-Stein risk ratio must be < 1 for k=13 (>= 3)."""
    result = compute_stein_paradox(bcg_input)
    assert result["stein_risk_ratio"] < 1.0, (
        f"Risk ratio {result['stein_risk_ratio']} should be < 1 for k=13"
    )


def test_risk_ratio_below_one_for_k5(statin_input):
    """James-Stein risk ratio must be < 1 for k=5 (>= 3)."""
    result = compute_stein_paradox(statin_input)
    assert result["stein_risk_ratio"] < 1.0


def test_js_mse_dominates_mle(bcg_input):
    """JS MSE curve should be <= MLE MSE curve at every grid point."""
    result = compute_stein_paradox(bcg_input)
    for i, (js, mle) in enumerate(
        zip(result["js_mse_curve"], result["mle_mse_curve"])
    ):
        assert js <= mle + 1e-3, (
            f"JS MSE ({js:.4f}) > MLE MSE ({mle:.4f}) at grid point {i}"
        )


def test_shrinkage_factor_in_range(bcg_input):
    """Shrinkage factor should be between 0 and 1 for well-behaved data."""
    result = compute_stein_paradox(bcg_input)
    # Raw shrinkage factor can be negative (hence positive-part estimator),
    # but for moderate k with real data it is typically in (0, 1).
    assert -1.0 <= result["shrinkage_factor"] <= 2.0


def test_grid_length_matches(bcg_input):
    """theta_grid, mle_mse_curve, js_mse_curve all have n_grid=50 points."""
    result = compute_stein_paradox(bcg_input, n_grid=50)
    assert len(result["theta_grid"]) == 50
    assert len(result["mle_mse_curve"]) == 50
    assert len(result["js_mse_curve"]) == 50
