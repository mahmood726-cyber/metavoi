"""Tests for Fisher Information module."""

import numpy as np
import pytest
from metavoi.fisher_information import compute_fisher_information


def test_fisher_matrix_positive_semidefinite(bcg_input):
    """Fisher information matrix must be PSD."""
    result = compute_fisher_information(bcg_input)
    mat = np.array(result["fisher_matrix"])
    eigenvalues = np.linalg.eigvalsh(mat)
    assert all(ev >= -1e-12 for ev in eigenvalues)


def test_effective_sample_size_equals_k(bcg_input):
    """With homogeneous vi = se^2, effective sample size should equal k."""
    result = compute_fisher_information(bcg_input)
    # n_eff = I_tt / I_single = (k/w) / (1/w) = k
    assert abs(result["effective_sample_size"] - bcg_input.k) < 1e-6


def test_information_ratio_between_zero_and_one(bcg_input):
    """Information ratio must be in [0, 1]."""
    result = compute_fisher_information(bcg_input)
    ratio = result["information_ratio"]
    assert 0.0 <= ratio <= 1.0


def test_det_fisher_positive(bcg_input):
    """Determinant of Fisher information must be positive."""
    result = compute_fisher_information(bcg_input)
    assert result["det_fisher"] > 0


def test_jeffreys_prior_grid_length(bcg_input):
    """Jeffreys prior should be evaluated on a 50-point grid."""
    result = compute_fisher_information(bcg_input)
    assert len(result["jeffreys_prior_tau2"]) == 50
    assert len(result["tau2_grid"]) == 50
    # Values must be positive
    assert all(v > 0 for v in result["jeffreys_prior_tau2"])
