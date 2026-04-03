import pytest
from metavoi.sensitivity_analysis import compute_sensitivity


def test_prcc_values_in_range(bcg_input):
    """All PRCC values must be in [-1, 1]."""
    result = compute_sensitivity(bcg_input, n_samples=500)
    for name, val in result["prcc"].items():
        assert -1.0 <= val <= 1.0, f"PRCC({name}) = {val} out of range"


def test_tornado_has_all_params(bcg_input):
    """Tornado diagram should include all four varied parameters."""
    result = compute_sensitivity(bcg_input, n_samples=500)
    param_names = {t["param"] for t in result["tornado"]}
    assert "theta" in param_names
    assert "tau2" in param_names
    assert "mcid" in param_names
    assert "population" in param_names


def test_tornado_range_nonnegative(bcg_input):
    """Tornado range values should be non-negative."""
    result = compute_sensitivity(bcg_input, n_samples=500)
    for t in result["tornado"]:
        assert t["range"] >= 0.0


def test_scatter_data_lengths_match(bcg_input):
    """Scatter plot data vectors should have same length as n_samples."""
    n = 500
    result = compute_sensitivity(bcg_input, n_samples=n)
    assert len(result["scatter_x1"]) == n
    assert len(result["scatter_y1"]) == n
    assert len(result["scatter_x2"]) == n
    assert len(result["scatter_y2"]) == n


def test_most_influential_is_valid(bcg_input):
    """Most influential parameter should be one of the varied parameters."""
    result = compute_sensitivity(bcg_input, n_samples=500)
    assert result["most_influential"] in ("theta", "tau2", "mcid", "population")
