"""Tests for Information-Theoretic VoI module."""

import pytest
from metavoi.entropy_voi import compute_entropy_voi


def test_decision_entropy_in_range(bcg_input):
    """Decision entropy must be in [0, 1] bits (binary decision)."""
    result = compute_entropy_voi(bcg_input, n_mc=3000)
    assert 0.0 <= result["decision_entropy"] <= 1.0


def test_mutual_info_equals_decision_entropy(bcg_input):
    """I(D; theta) = H(D) since H(D|theta) = 0 with perfect info."""
    result = compute_entropy_voi(bcg_input, n_mc=3000)
    assert abs(result["mutual_information_theta"] - result["decision_entropy"]) < 1e-10


def test_entropy_reduction_nonnegative(bcg_input):
    """Entropy reduction from trial should be non-negative for all sizes."""
    result = compute_entropy_voi(bcg_input, n_mc=3000)
    for point in result["entropy_reduction_curve"]:
        assert point["delta_h"] >= 0.0, f"Negative entropy reduction at n={point['n']}"


def test_kl_gain_nonnegative(bcg_input):
    """KL divergence (information gain) must be non-negative."""
    result = compute_entropy_voi(bcg_input, n_mc=3000)
    for point in result["kl_gain_curve"]:
        assert point["kl"] >= 0.0, f"Negative KL at n={point['n']}"


def test_capacity_utilization_bounded(bcg_input):
    """Channel capacity utilization must be in [0, 1]."""
    result = compute_entropy_voi(bcg_input, n_mc=3000)
    assert 0.0 <= result["capacity_utilization"] <= 1.0
