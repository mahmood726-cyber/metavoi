"""Tests for multi-alternative decision VoI."""

import pytest
from metavoi.multi_decision import Alternative, compute_multi_evpi
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_two_arms_matches_binary_evpi(bcg_input):
    """With 2 arms (treat vs no-treat), multi EVPI ~ binary EVPI."""
    # Binary EVPI from existing code
    draws = predictive_distribution(bcg_input)
    evpi_binary = compute_evpi(draws, bcg_input.mcid)

    # Multi-arm with 2 alternatives: treat (effect=theta) and no-treat (effect=0)
    # Treat arm uses predictive distribution (se^2 + tau2) to match binary EVPI
    alts = [
        Alternative(label="treat", effect=bcg_input.theta, se=bcg_input.se,
                    cost=0.0, tau2=bcg_input.tau2),
        Alternative(label="no_treat", effect=0.0, se=1e-8, cost=0.0),
    ]
    result = compute_multi_evpi(
        alts, bcg_input.mcid, population=bcg_input.population,
        n_sim=bcg_input.n_sim, seed=bcg_input.seed,
    )

    # Should be in the same order of magnitude
    if evpi_binary > 0.001:
        ratio = result["evpi_multi"] / evpi_binary
        assert 0.2 <= ratio <= 5.0, (
            f"Multi={result['evpi_multi']:.4f} vs Binary={evpi_binary:.4f}"
        )


def test_multi_evpi_nonnegative(bcg_input):
    """EVPI must be non-negative."""
    alts = [
        Alternative("A", -0.5, 0.15, 100),
        Alternative("B", -0.3, 0.10, 200),
        Alternative("C", -0.1, 0.20, 50),
    ]
    result = compute_multi_evpi(alts, bcg_input.mcid, seed=bcg_input.seed)
    assert result["evpi_multi"] >= 0.0


def test_p_optimal_sums_to_one(bcg_input):
    """Probabilities of being optimal must sum to 1.0."""
    alts = [
        Alternative("A", -0.5, 0.15, 100),
        Alternative("B", -0.3, 0.10, 200),
        Alternative("C", -0.1, 0.20, 50),
    ]
    result = compute_multi_evpi(alts, bcg_input.mcid, seed=bcg_input.seed)
    assert abs(sum(result["p_optimal"]) - 1.0) < 1e-6


def test_pairwise_evpi_symmetric(bcg_input):
    """Pairwise EVPI matrix must be symmetric."""
    alts = [
        Alternative("A", -0.5, 0.15, 100),
        Alternative("B", -0.3, 0.10, 200),
    ]
    result = compute_multi_evpi(alts, bcg_input.mcid, seed=bcg_input.seed)
    pw = result["pairwise_evpi"]
    K = len(alts)
    for i in range(K):
        for j in range(K):
            assert abs(pw[i][j] - pw[j][i]) < 1e-10


def test_empty_alternatives():
    """Zero alternatives returns graceful empty result."""
    result = compute_multi_evpi([], mcid=-0.2)
    assert result["evpi_multi"] == 0.0
    assert result["p_optimal"] == []
    assert result["current_best"] == ""
