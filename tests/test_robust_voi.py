"""Tests for Distributionally Robust VoI module."""

import pytest
from metavoi.robust_voi import compute_robust_voi


def test_robust_evpi_geq_nominal(bcg_input):
    """Wasserstein robust EVPI must be >= nominal EVPI for all eps."""
    result = compute_robust_voi(bcg_input)
    nominal = result["nominal_evpi"]
    for point in result["robust_evpi_curve"]:
        assert point["robust_evpi"] >= nominal - 1e-12, (
            f"Robust EVPI {point['robust_evpi']} < nominal {nominal} at eps={point['eps']}"
        )


def test_breakeven_epsilon_positive(bcg_input):
    """Breakeven epsilon must be positive when effect is clearly beneficial."""
    result = compute_robust_voi(bcg_input)
    assert result["breakeven_epsilon"] > 0.0


def test_chebyshev_p_wrong_bounded(bcg_input):
    """Chebyshev worst-case P(wrong) must be in [0, 1]."""
    result = compute_robust_voi(bcg_input)
    assert 0.0 <= result["chebyshev_p_wrong"] <= 1.0


def test_chebyshev_evpi_nonneg(bcg_input):
    """Chebyshev EVPI must be non-negative."""
    result = compute_robust_voi(bcg_input)
    assert result["chebyshev_evpi"] >= 0.0


def test_contamination_monotone(bcg_input):
    """Contamination robust EVPI should be monotonically non-decreasing in eps."""
    result = compute_robust_voi(bcg_input)
    curve = result["contamination_curve"]
    for i in range(1, len(curve)):
        assert curve[i]["contamination_evpi"] >= curve[i - 1]["contamination_evpi"] - 1e-12, (
            f"Contamination EVPI decreased from eps={curve[i-1]['eps']} to eps={curve[i]['eps']}"
        )
