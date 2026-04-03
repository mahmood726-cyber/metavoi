"""Tests for Causal Value of Information module."""

import pytest
from metavoi.causal_voi import compute_causal_voi


def test_e_value_positive(bcg_input):
    """E-value must be positive for any non-zero effect."""
    result = compute_causal_voi(bcg_input)
    assert result["e_value"] > 0.0


def test_causal_evpi_geq_standard(bcg_input):
    """Causal EVPI (with confounding) should be >= standard EVPI."""
    result = compute_causal_voi(bcg_input)
    # Confounding adds uncertainty, so causal EVPI >= standard (with MC tolerance)
    assert result["causal_evpi"] >= result["standard_evpi"] - 0.01


def test_confounding_component_sign(bcg_input):
    """Confounding component should be non-negative (confounding adds uncertainty)."""
    result = compute_causal_voi(bcg_input)
    # Allow small negative due to MC noise
    assert result["confounding_component"] >= -0.01


def test_iv_curve_has_entries(bcg_input):
    """IV-EVPI curve must have entries for F in [1, 2, 5, 10]."""
    result = compute_causal_voi(bcg_input)
    f_values = [pt["F"] for pt in result["iv_evpi_curve"]]
    assert f_values == [1, 2, 5, 10]


def test_is_causally_robust_type(bcg_input):
    """is_causally_robust must be a boolean."""
    result = compute_causal_voi(bcg_input)
    assert isinstance(result["is_causally_robust"], bool)
