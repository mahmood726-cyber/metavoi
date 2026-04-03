import pytest
from metavoi.importance_evsi import (
    compute_importance_evsi,
    compute_importance_evsi_curve,
    compute_comparison,
)
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_importance_evsi_nonnegative(bcg_input):
    """Importance-sampling EVSI must be non-negative."""
    evsi = compute_importance_evsi(bcg_input, n_trial=500)
    assert evsi >= 0.0


def test_importance_evsi_increases_with_n(bcg_input):
    """EVSI should generally increase with trial size (allowing MC noise)."""
    evsi_100 = compute_importance_evsi(bcg_input, n_trial=100)
    evsi_2000 = compute_importance_evsi(bcg_input, n_trial=2000)
    assert evsi_2000 >= evsi_100 - 0.02  # MC noise tolerance


def test_importance_evsi_bounded_by_evpi(bcg_input):
    """EVSI should not exceed EVPI (plus MC tolerance)."""
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    evsi = compute_importance_evsi(bcg_input, n_trial=5000)
    assert evsi <= evpi + 0.01


def test_comparison_curve_lengths(bcg_input):
    """Comparison should return matching-length curves."""
    result = compute_comparison(bcg_input, n_values=[100, 500, 1000])
    assert len(result["evsi_curve"]) == 3
    assert len(result["evsi_moment_curve"]) == 3
    assert len(result["efficiency_ratio"]) == 3


def test_recommended_n_is_positive(bcg_input):
    """Recommended N should be a positive integer from the evaluated set."""
    result = compute_comparison(bcg_input, n_values=[100, 500, 1000, 2000])
    assert result["recommended_n"] in [100, 500, 1000, 2000]
