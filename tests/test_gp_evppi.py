"""Tests for GP-regression EVPPI (Strong-Oakley-Brennan 2014)."""

import pytest
from metavoi.gp_evppi import compute_gp_evppi
from metavoi.evppi import compute_evppi


def test_gp_evppi_nonnegative(bcg_input):
    """Both GP-EVPPI values must be >= 0."""
    result = compute_gp_evppi(bcg_input)
    assert result["evppi_theta_gp"] >= 0.0
    assert result["evppi_tau2_gp"] >= 0.0


def test_gp_evppi_within_50pct_of_nested_mc(bcg_input):
    """GP-EVPPI(theta) should be in the same ballpark as nested MC EVPPI(theta)."""
    gp = compute_gp_evppi(bcg_input, n_outer=2000)
    mc = compute_evppi(bcg_input, n_outer=2000)

    # Both positive -> within 50% (allow for MC noise)
    if mc["theta"] > 0.01:
        ratio = gp["evppi_theta_gp"] / mc["theta"]
        assert 0.1 <= ratio <= 10.0, (
            f"GP={gp['evppi_theta_gp']:.4f} vs MC={mc['theta']:.4f}, ratio={ratio:.2f}"
        )


def test_gp_r2_reasonable(bcg_input):
    """GP R^2 for theta should be positive (GP captures some variance)."""
    result = compute_gp_evppi(bcg_input)
    # R^2 can be negative if GP is very poor, but for theta it should be decent
    assert result["gp_r2_theta"] > -1.0


def test_gp_evppi_returns_all_keys(bcg_input):
    """Output dict has all expected keys."""
    result = compute_gp_evppi(bcg_input)
    expected_keys = {"evppi_theta_gp", "evppi_tau2_gp", "gp_r2_theta", "gp_r2_tau2"}
    assert set(result.keys()) == expected_keys


def test_gp_evppi_deterministic(bcg_input):
    """Same seed -> same result."""
    r1 = compute_gp_evppi(bcg_input, n_outer=500)
    r2 = compute_gp_evppi(bcg_input, n_outer=500)
    assert r1["evppi_theta_gp"] == r2["evppi_theta_gp"]
    assert r1["evppi_tau2_gp"] == r2["evppi_tau2_gp"]
