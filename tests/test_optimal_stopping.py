"""Tests for Optimal Stopping module."""

import math
import pytest
from metavoi.optimal_stopping import compute_optimal_stopping


def test_secretary_threshold_approx_k_over_e(bcg_input):
    """Secretary threshold should be ceil(k/e) or close."""
    result = compute_optimal_stopping(bcg_input)
    expected = max(1, int(round(bcg_input.k / math.e)))
    assert result["secretary_threshold"] == expected


def test_sprt_boundaries_finite(bcg_input):
    """SPRT upper and lower boundaries must be finite and correctly ordered."""
    result = compute_optimal_stopping(bcg_input)
    upper = result["sprt_boundaries"]["upper"]
    lower = result["sprt_boundaries"]["lower"]
    assert math.isfinite(upper), f"Upper boundary not finite: {upper}"
    assert math.isfinite(lower), f"Lower boundary not finite: {lower}"
    assert upper > lower, f"Upper ({upper}) should exceed lower ({lower})"


def test_sprt_expected_n_positive(statin_input):
    """Expected sample sizes under H0 and H1 must be positive."""
    result = compute_optimal_stopping(statin_input)
    assert result["sprt_expected_n_h0"] > 0
    assert result["sprt_expected_n_h1"] > 0


def test_cusum_arl_positive(bcg_input):
    """CUSUM average run length must be positive."""
    result = compute_optimal_stopping(bcg_input)
    assert result["cusum_arl"] > 0


def test_value_of_continuing_nonnegative(uncertain_input):
    """Value of continuing should be >= 0 (option value can't be negative)."""
    result = compute_optimal_stopping(uncertain_input)
    assert result["value_of_continuing"] >= 0.0
