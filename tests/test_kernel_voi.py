"""Tests for Kernel Methods VoI module."""

import pytest
from metavoi.kernel_voi import compute_kernel_voi


def test_mmd_nonnegative(bcg_input):
    """MMD values must be non-negative for all trial sizes."""
    result = compute_kernel_voi(bcg_input, n_mc=1500)
    for point in result["mmd_curve"]:
        assert point["mmd"] >= 0.0, f"MMD negative at n={point['n']}"


def test_bandwidth_positive(bcg_input):
    """Kernel bandwidth from median heuristic must be positive."""
    result = compute_kernel_voi(bcg_input, n_mc=1500)
    assert result["bandwidth"] > 0.0


def test_kernel_evppi_nonnegative(bcg_input):
    """Kernel EVPPI for theta and tau2 must be non-negative."""
    result = compute_kernel_voi(bcg_input, n_mc=1500)
    assert result["kernel_evppi_theta"] >= 0.0
    assert result["kernel_evppi_tau2"] >= 0.0


def test_two_sample_p_value_valid(bcg_input):
    """Two-sample permutation p-value must be in (0, 1]."""
    result = compute_kernel_voi(bcg_input, n_mc=1500)
    assert 0.0 < result["two_sample_p_value"] <= 1.0


def test_kernel_mean_norm_positive(bcg_input):
    """Kernel mean embedding norm must be positive (non-degenerate kernel)."""
    result = compute_kernel_voi(bcg_input, n_mc=1500)
    assert result["kernel_mean_norm"] > 0.0
