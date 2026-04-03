import pytest
from metavoi.regret import compute_regret
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_regret_opportunity_loss_approx_evpi(bcg_input):
    """Opportunity loss should approximate EVPI (within 20%)."""
    result = compute_regret(bcg_input)
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    # Both are MC estimates — allow 20% tolerance
    assert abs(result["opportunity_loss"] - evpi) < 0.20 * evpi + 0.005


def test_regret_surfaces_nonnegative(bcg_input):
    """Regret surfaces must be non-negative everywhere."""
    result = compute_regret(bcg_input)
    assert all(r >= -1e-12 for r in result["regret_surface_treat"])
    assert all(r >= -1e-12 for r in result["regret_surface_no_treat"])


def test_minimax_decision_is_string(bcg_input):
    """Minimax decision should be treat or no_treat."""
    result = compute_regret(bcg_input)
    assert result["minimax_decision"] in ("treat", "no_treat")


def test_theta_grid_has_100_points(bcg_input):
    """Theta grid should have exactly 100 points."""
    result = compute_regret(bcg_input)
    assert len(result["theta_grid"]) == 100
    assert len(result["regret_surface_treat"]) == 100


def test_regret_optimal_n_is_positive(bcg_input):
    """Regret-optimal N should be a positive integer."""
    result = compute_regret(bcg_input)
    assert result["regret_optimal_n"] is not None
    assert result["regret_optimal_n"] > 0
